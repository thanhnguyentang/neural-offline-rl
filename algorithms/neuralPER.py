import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm 
from .utils import Model, inv_sherman_morrison, cls2bandit_context
from functorch import make_functional_with_buffers, vmap, grad, jacrev

class NeuralPER(object):
    def __init__(self,
                 bandit,
                 is_cls_bandit=True, 
                 T = 10000, 
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
                 perturbed_variance = 1, 
                 M=1, 
                 evaluate_every=1,
                 sample_window=1000
                #  start_train_after = 1
                 ):

        self.is_cls_bandit = is_cls_bandit
        self.bandit = bandit 
        self.sample_window = sample_window
        # hidden size of the NN layers
        self.hidden_size = hidden_size
        # number of layers
        self.n_layers = n_layers

        # number of rewards in the training buffer
        self.batch_size = batch_size

        # self.start_train_after = start_train_after
        self.M = M # number of bootstrapps 

        self.T = T
        self.num_test = self.bandit.num_test

        # NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_every = train_every 
        self.throttle = throttle
        self.reg_factor = reg_factor

        self.evaluate_every = evaluate_every

        self.perturbed_variance = perturbed_variance 

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')


        # dropout rate
        self.p = p   

        # create an ensemble 
        self.ensemble = [ Model(input_size=bandit.context_dim * self.bandit.n_arms if self.is_cls_bandit else bandit.context_dim,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device) for _ in range(self.M) ] 

        self.optimizers = [torch.optim.Adam(self.ensemble[m].parameters(), lr=self.learning_rate) for m in range(self.M) ]

        self.reset()


    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    def evaluate(self):
        """
        """
        [self.ensemble[m].eval() for m in range(self.M)] 
        ensemble_preds = np.zeros((self.M, self.num_test, self.bandit.n_arms))
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
            for m in range(self.M):
                ensemble_preds[m,:,a] = self.ensemble[m].forward(x_batch).detach().squeeze().cpu().detach().numpy()
       
        # predicted_arms = np.argmax(np.min(ensemble_preds, axis=0 ), axis=1).astype('int') #(n,)
        # subopts = self.bandit.best_rewards[-self.num_test:] - self.bandit.rewards[-self.num_test:, :][np.arange(self.num_test), predicted_arms]
        # subopt = np.mean(subopts) # estimated expected sub-optimality
        # self.subopts.append((self.iteration, subopt)) #save the index of offline data and the curresponding regret
        # opt_arm_select_percent = np.mean(predicted_arms == self.bandit.best_arms[-self.num_test:])
        # self.opt_arm_select_percent_stats.append((self.iteration, opt_arm_select_percent))

        preds = np.min(ensemble_preds, axis=0 )
        opt_pred_sel = np.array(preds - np.max(preds, axis=1)[:,None] == 0).astype('float') # (n,a)
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
        self.iteration = 0
        self.update_times = [] # collect the update time 
        self.action_selection_times = [] # collect the action selection time 
        self.opt_arm_select_percent_stats = []
        self.subopts = []


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


        sample_pool = np.arange(self.iteration)[-self.sample_window:] # force to the latest samples only 
        pool_size = sample_pool.shape[0] 

        for m in range(self.M): 
            y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, offline_actions_so_far]).squeeze().to(self.device)
            y_train = y_train + self.perturbed_variance * torch.randn(y_train.shape).to(self.device)

            # train mode
            self.ensemble[m].train()
            for _ in range(self.epochs):
                # rand_indices = np.random.choice(self.iteration, size=self.batch_size, replace=True)
                rand_indices = np.random.choice(sample_pool, size=self.batch_size, replace=True if self.batch_size >= pool_size else False)
                x_batch = x_train[rand_indices] 
                y_batch = y_train[rand_indices] 
                y_pred = self.ensemble[m](x_batch).squeeze()
                loss = nn.MSELoss()(y_batch, y_pred) 
                self.optimizers[m].zero_grad()
                loss.backward()
                self.optimizers[m].step()

    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'update time': -1,
        }

        with tqdm(total=self.T, postfix=postfix) as pbar:
            for t in range(self.T):
                self.offline_action = self.bandit.offline_arms[t] # Get offline action for updating the internal model 
                
                # update approximator
                if t % self.train_every == 0 and t >= self.bandit.n_arms:
                    train_start = time.time()
                    self.train()
                    train_end = time.time() 
                    self.update_times.append((t, train_end - train_start)) # @REMARK: not (yet) divided by M

                # Evaluate 

                if t % self.evaluate_every == 0: 
                    start_time = time.time()
                    self.evaluate() # include computing grad_out and A_inv in the test data
                    end_time = time.time()
                    elapsed_time_per_arm = (end_time - start_time) / self.bandit.n_arms # @REMARK: not (yet) divided by M 
                    self.action_selection_times.append((t, elapsed_time_per_arm ))
                    # action selection time include computing A_inv, grad_out, and test_grad_out

                    print('\n[NeuralPER] t={}, subopt={}, % optimal arms={}, action select time={}'.format(t, self.subopts[-1][1], \
                        self.opt_arm_select_percent_stats[-1][1], self.action_selection_times[-1][1] ))

                # increment counter
                self.iteration += 1

                # log
                update_time = sum([ item[1] for item in self.update_times]) / len(self.update_times) if len(self.update_times) > 0 else 0 
                postfix['update time'] = '{}'.format(update_time)


                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)