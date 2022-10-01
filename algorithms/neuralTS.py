import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm 
from .utils import Model, inv_sherman_morrison, cls2bandit_context

class NeuralTS(object):
    """Neural TS.
    """
    def __init__(self,
                 bandit,
                 hidden_size=20,
                 n_layers=2,
                 reg_factor=1.0,
                 batch_size=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 train_every=1,
                 throttle=1,
                 use_cuda=False,
                 scaled_variance = 1, # nu in NeuralTS paper 
                #  start_train_after = 1
                 ):

        self.bandit = bandit 
        # hidden size of the NN layers
        self.hidden_size = hidden_size
        # number of layers
        self.n_layers = n_layers

        # number of rewards in the training buffer
        self.batch_size = batch_size
        # self.start_train_after = start_train_after

        # NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_every = train_every 
        self.throttle = throttle
        self.reg_factor = reg_factor

        self.scaled_variance = scaled_variance 

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        # dropout rate
        self.p = p

        # neural network
        self.model = Model(input_size=bandit.context_dim,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # maximum L2 norm for the features across all arms and all rounds
        self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))

        self.reset()

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights.
        """
        for a in self.bandit.arms:
            x = self.bandit.features[self.iteration] # cls context 
            a_hot = np.zeros(self.bandit.n_arms) 
            a_hot[a] = 1 
            x = np.kron(a_hot, x)
            # convert cls 
            x = torch.FloatTensor(x.reshape(1, -1)).to(self.device)

            self.model.zero_grad()
            y = self.model(x)
            y.backward()

            self.grad_approx[a] = torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if w.requires_grad]
            ).cpu().detach().numpy()

    def update_A_inv(self):
        self.A_inv[self.action] = inv_sherman_morrison(
            self.grad_approx[self.action],
            self.A_inv[self.action]
        )

    def reset(self):
        """Reset the internal estimates.
        """
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = np.array(
            [
                np.eye(self.approximator_dim)/self.reg_factor for _ in self.bandit.arms
            ]
        )

    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def reset_regrets(self):
        """Initialize regrets.
        """
        self.regrets = np.empty(self.bandit.T)

    def reset_actions(self):
        """Initialize cache of actions.
        """
        self.actions = np.empty(self.bandit.T).astype('int')

    def train(self):
        """Train neural approximator.
        """
        iterations_so_far = range(self.iteration+1)
        actions_so_far = self.actions[:self.iteration+1]

        x_train = self.bandit.features[iterations_so_far] 
        x_train = cls2bandit_context(x_train, actions_so_far, self.bandit.n_arms)
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, actions_so_far]).squeeze().to(self.device)

        # train mode
        self.model.train()
        for _ in range(self.epochs):
            rand_indices = np.random.choice(self.iteration, size=self.batch_size, replace=True)
            x_batch = x_train[rand_indices] 
            y_batch = y_train[rand_indices] 
            y_pred = self.model.forward(x_batch).squeeze()
            loss = nn.MSELoss()(y_batch, y_pred) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def sample_action(self):
        """Return the action to play based on current estimates
        """
        # Compute the covariance matrix, make sure to call update_grad_approx before.
        if self.iteration < self.bandit.n_arms: 
            return self.iteration
        else:
            x = self.bandit.features[self.iteration] # (d) 
            a_hot = np.eye(self.bandit.n_arms) # (n_arms, n_arms)
            x = torch.Tensor(np.kron(a_hot, x[None,:])).to(self.device) # (n_arms, d*n_arms)
            self.model.eval()
            mus = self.model(x)
            r = np.zeros(self.bandit.n_arms)
            for a in range(self.bandit.n_arms):
                sigma = self.scaled_variance * np.sqrt(np.dot(self.grad_approx[a], np.dot(self.A_inv[a], self.grad_approx[a].T)))
                r[a] = sigma * np.random.randn() + mus[a]

            return np.argmax(r).astype('int')

    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'total regret': 0.0,
            '% optimal arm': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):
                # update confidence of all arms based on observed features at time t
                self.update_output_gradient()
                self.action = self.sample_action()
                self.actions[t] = self.action
                if t % self.train_every == 0 and t >= self.bandit.n_arms:
                    self.train()
                self.update_A_inv()
                self.regrets[t] = self.bandit.best_rewards[t]-self.bandit.rewards[t, self.action]
                self.iteration += 1

                # log
                postfix['total regret'] += self.regrets[t]
                n_optimal_arm = np.sum(
                    self.actions[:self.iteration] == self.bandit.best_arms[:self.iteration]
                )
                postfix['% optimal arm'] = '{:.2%}'.format(n_optimal_arm / self.iteration)

                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)
