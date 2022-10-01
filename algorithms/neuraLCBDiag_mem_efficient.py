import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm 
from .utils import Model, inv_sherman_morrison, cls2bandit_context
from functorch import make_functional_with_buffers, vmap, grad, jacrev

class NeuraLCBDiag(object):
    """NeuraLCB.
    """
    def __init__(self,
                 bandit,
                 T = 10000, 
                 hidden_size=20,
                 n_layers=2,
                 reg_factor=1.0,
                 beta=1,
                 batch_size=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 train_every=1,
                 evaluate_every=1, 
                 throttle=1,
                 use_cuda=False,
                 is_cls_bandit = True,
                 sample_window=1000
                 ):

        self.is_cls_bandit = is_cls_bandit
        self.sample_window = sample_window 

        self.bandit = bandit 
        self.num_test = self.bandit.num_test
        # self.test_bandit = test_bandit 
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
        self.reg_factor = reg_factor 
        self.train_every = train_every 
        self.evaluate_every = evaluate_every
        self.throttle = throttle 
        self.beta = beta 

        self.T = T 

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        # dropout rate
        self.p = p

        # neural network
        self.model = Model(input_size=bandit.context_dim * self.bandit.n_arms if self.is_cls_bandit else bandit.context_dim, 
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.reset()


    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights on the test contexts only 
        """
        for a in self.bandit.arms:
            if self.is_cls_bandit:
                x = self.bandit.features[self.iteration] # cls context 
                a_hot = np.zeros(self.bandit.n_arms) 
                a_hot[a] = 1 
                x = np.kron(a_hot, x)
            else: 
                x = self.bandit.features[self.iteration, a]
            # convert cls 
            x = torch.FloatTensor(x.reshape(1, -1)).to(self.device)

            self.model.zero_grad()
            y = self.model(x)
            y.backward()

            self.grad_approx[a] = torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if w.requires_grad]
            ).cpu().detach().numpy()


    def evaluate(self):
        """
        """
        self.model.eval()
        lower_confidence_bounds = np.zeros((self.num_test, self.bandit.n_arms))
        for a in tqdm(self.bandit.arms):
            if self.is_cls_bandit:
                x_batch = self.bandit.features[-self.num_test:, :] # (n,d)
                a_hot = np.zeros(self.bandit.n_arms) 
                a_hot[a] = 1 
                x_batch = np.kron(a_hot, x_batch) # (n, da)
            else: 
                x_batch = self.bandit.features[-self.num_test:, a] # (n,d)
            # convert cls 
            x_batch = torch.FloatTensor(x_batch).to(self.device)
            func, params, buffers = make_functional_with_buffers(self.model)
            test_grad_approx = vmap(jacrev(func), (None, None, 0))(params, buffers, x_batch)

            test_grad_approx = torch.cat(
                [w.reshape(self.num_test, -1) / np.sqrt(self.hidden_size) for w in test_grad_approx], dim=1
            ).cpu().detach().numpy() # (n, D)

            A_inv_diag = 1 / self.A_diag[a] # (D,)
            exploration_bonus = self.beta * np.sqrt( np.sum( (test_grad_approx * A_inv_diag[None,:]) * test_grad_approx, axis=1) )
            # exploration_bonus = self.beta * np.sqrt(np.sum((test_grad_approx @ self.A_inv[a]) * test_grad_approx, axis=1)) # (n,)
            mu_hat = self.model.forward(x_batch).detach().squeeze().cpu().detach().numpy()
            lower_confidence_bounds[:,a] = mu_hat - exploration_bonus # (n,)
       
        # predicted_arms = np.argmax(lower_confidence_bounds, axis=1).astype('int')
        # subopts = self.bandit.best_rewards[-self.num_test:] - self.bandit.rewards[-self.num_test:, :][np.arange(self.num_test), predicted_arms]
        # subopt = np.mean(subopts) # estimated expected sub-optimality
        # self.subopts.append((self.iteration, subopt)) #save the index of offline data and the curresponding regret
        # opt_arm_select_percent = np.mean(predicted_arms == self.bandit.best_arms[-self.num_test:])
        # self.opt_arm_select_percent_stats.append((self.iteration, opt_arm_select_percent))

        opt_pred_sel = np.array(lower_confidence_bounds - np.max(lower_confidence_bounds, axis=1)[:,None] == 0).astype('float') # (n,a)
        subopts = self.bandit.best_rewards[-self.num_test:] - np.sum(self.bandit.rewards[-self.num_test:, :] * opt_pred_sel, axis=1) / np.sum(opt_pred_sel, axis=1)[:,None]        
        subopt = np.mean(subopts) # estimated expected sub-optimality
        self.subopts.append((self.iteration, subopt)) #save the index of offline data and the curresponding regret
        # new eval of opt-rate
        best_arms = np.zeros((self.num_test, self.bandit.n_arms)) 
        best_arms[np.arange(self.num_test), self.bandit.best_arms[-self.num_test:].astype('int')] = 1 
        opt_arm_select_percent = np.mean(np.sum(best_arms * opt_pred_sel, axis=1) / np.sum(opt_pred_sel, axis=1))
        self.opt_arm_select_percent_stats.append((self.iteration, opt_arm_select_percent))
        
        
    def reset(self):
        """Reset the internal estimates.
        """
        # self.reset_A_inv()
        self.reset_A_diag()
        self.reset_grad_approx()
        self.iteration = 0
        self.update_times = [] # collect the update time 
        self.action_selection_times = [] # collect the action selection time 
        self.opt_arm_select_percent_stats = []
        self.subopts = []

    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = np.array(
            [
                np.eye(self.approximator_dim)/self.reg_factor for _ in self.bandit.arms
            ]
        )

    def reset_A_diag(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_diag = self.reg_factor * np.ones((self.bandit.n_arms, self.approximator_dim))


    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def update_A_inv(self): # update on the stream of offline actions from the train contexts
        A_diag = 1. / np.diag(self.A_inv[self.offline_action])
        u = self.grad_approx[self.offline_action] 
        A_diag = A_diag + np.square(u)
        self.A_inv[self.offline_action] = np.diag(1. / A_diag)

    def update_A_diag(self): # update on the stream of offline actions from the train contexts
        self.A_diag[self.offline_action] = self.A_diag[self.offline_action] + np.square(self.grad_approx[self.offline_action] )

    def train(self):
        """Train neural approximator.
        """
        iterations_so_far = range(self.iteration+1)
        # actions_so_far = self.actions[:self.iteration+1]
        offline_actions_so_far = self.bandit.offline_arms[:self.iteration+1]

        if self.is_cls_bandit:
            x_train = self.bandit.features[iterations_so_far] 
            x_train = cls2bandit_context(x_train, offline_actions_so_far, self.bandit.n_arms)
        else:
            x_train = self.bandit.features[iterations_so_far, offline_actions_so_far]
    
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, offline_actions_so_far]).squeeze().to(self.device)

        # train mode
        self.model.train()
        # loss_ = 0
        sample_pool = np.arange(self.iteration)[-self.sample_window:] # force to the latest samples only 
        pool_size = sample_pool.shape[0] 
        assert pool_size == min(self.sample_window, self.iteration)

        for _ in range(self.epochs):
            # rand_indices = np.random.choice(self.iteration, size=self.batch_size, replace=True)
            rand_indices = np.random.choice(sample_pool, size=self.batch_size, replace=True if self.batch_size >= pool_size else False)
            x_batch = x_train[rand_indices] 
            y_batch = y_train[rand_indices] 
            y_pred = self.model.forward(x_batch).squeeze()
            loss = nn.MSELoss()(y_batch, y_pred) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'update time': -1,
        }

        inv_A_times = []

        with tqdm(total=self.T, postfix=postfix) as pbar:
            for t in range(self.T):
                self.offline_action = self.bandit.offline_arms[t] # Get offline action for updating the internal model 
                
                # update approximator
                if t % self.train_every == 0 and t >= self.bandit.n_arms:
                    train_start = time.time()
                    self.train()
                    train_end = time.time() 
                    self.update_times.append((t, train_end - train_start))

                # Update the offline grad_out and A_inv

                start_time = time.time() 
                self.update_output_gradient()
                # self.update_A_inv()
                self.update_A_diag()
                end_time = time.time() 
                inv_A_times.append(end_time - start_time)

                # Evaluate 

                if t % self.evaluate_every == 0: 
                    start_time = time.time()
                    self.evaluate() # include computing grad_out and A_inv in the test data
                    end_time = time.time()
                    elapsed_time_per_arm = (end_time - start_time) / self.bandit.n_arms
                    self.action_selection_times.append((t, elapsed_time_per_arm + sum(inv_A_times) ))
                    # action selection time include computing A_inv, grad_out, and test_grad_out

                    print('\n[NeuraLCBDiag] t={}, subopt={}, % optimal arms={}, action select time={}'.format(t, self.subopts[-1][1], \
                        self.opt_arm_select_percent_stats[-1][1], self.action_selection_times[-1][1] ))

                # increment counter
                self.iteration += 1

                # log
                update_time = sum([ item[1] for item in self.update_times]) / len(self.update_times) if len(self.update_times) > 0 else 0 
                postfix['update time'] = '{}'.format(update_time)


                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)