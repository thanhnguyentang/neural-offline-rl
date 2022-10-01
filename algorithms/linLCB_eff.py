"""
phi = x instead of hot_vec(x) to save mem for large data
"""
import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm 
from .utils import Model, inv_sherman_morrison, cls2bandit_context

class LinLCB(object):
    def __init__(self,
                bandit,
                is_cls_bandit=True, 
                evaluate_every = 1, 
                T = 10000, 
                reg_factor=1.0,
                beta=1,
                throttle=1
                ):

        self.is_cls_bandit=is_cls_bandit
        self.bandit = bandit 
        self.evaluate_every = evaluate_every
        self.num_test = self.bandit.num_test

        # NN parameters
        self.reg_factor = reg_factor 
        self.throttle = throttle 
        self.beta = beta 

        self.T = T 

        self.reset()


    def evaluate(self):
        """
        """
        print(self.A_inv.shape)
        lower_confidence_bounds = np.zeros((self.num_test, self.bandit.n_arms)) # (n,a)
        for a in tqdm(self.bandit.arms):
            if self.is_cls_bandit:
                x_batch = self.bandit.features[-self.num_test:, :] # (n,d)
                # a_hot = np.zeros(self.bandit.n_arms) 
                # a_hot[a] = 1 
                # x_batch = np.kron(a_hot, x_batch) # (n, da)
            else: 
                x_batch = self.bandit.features[-self.num_test:, a] # (n,d)
            # convert cls 

            exploration_bonus = self.beta * np.sqrt(np.sum((x_batch @ self.A_inv[a]) * x_batch, axis=1)) # (n,)
            mu_hat = np.dot(x_batch, self.theta_hat[a,:])
            lower_confidence_bounds[:,a] = mu_hat - exploration_bonus # (n,)
       
        predicted_arms = np.argmax(lower_confidence_bounds, axis=1).astype('int')
        
        # new evaluation: take into account multiple predicted best arms (prevent the algo from always selecting arm 0 when value fn is constant)
        opt_pred_sel = np.array(lower_confidence_bounds - np.max(lower_confidence_bounds, axis=1)[:,None] == 0).astype('float') # (n,a)
        # print(opt_pred_sel)
        subopts = self.bandit.best_rewards[-self.num_test:] - np.sum(self.bandit.rewards[-self.num_test:, :] * opt_pred_sel, axis=1) / np.sum(opt_pred_sel, axis=1)[:,None]

        # old eval of subopt
        # subopts = self.bandit.best_rewards[-self.num_test:] - self.bandit.rewards[-self.num_test:, :][np.arange(self.num_test), predicted_arms]
        
        subopt = np.mean(subopts) # estimated expected sub-optimality

        self.subopts.append((self.iteration, subopt)) #save the index of offline data and the curresponding regret

        # old eval of opt-rate
        # opt_arm_select_percent = np.mean(predicted_arms == self.bandit.best_arms[-self.num_test:])

        # new eval of opt-rate
        best_arms = np.zeros((self.num_test, self.bandit.n_arms)) 
        best_arms[np.arange(self.num_test), self.bandit.best_arms[-self.num_test:].astype('int')] = 1 
        opt_arm_select_percent = np.mean(np.sum(best_arms * opt_pred_sel, axis=1) / np.sum(opt_pred_sel, axis=1))


        self.opt_arm_select_percent_stats.append((self.iteration, opt_arm_select_percent))
        
        
    def reset(self):
        """Reset the internal estimates.
        """
        if self.is_cls_bandit:
            self.approximator_dim =  self.bandit.features[0].shape[0] 
        else: 
            self.approximator_dim =  self.bandit.features[0,0].shape[0] 
        self.offline_action = 0

        self.reset_A_inv()
        self.reset_phi()
        self.reset_y_hat()
        self.reset_theta_hat()
        self.iteration = 0
        self.update_times = [(0,0)] # collect the update time 
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

    def reset_phi(self):
        self.phi = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def reset_theta_hat(self): 
        self.theta_hat = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def reset_y_hat(self):
        self.y_hat = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def update_phi(self):
        for a in self.bandit.arms:
            if self.is_cls_bandit:
                x = self.bandit.features[self.iteration] # cls context 
                # a_hot = np.zeros(self.bandit.n_arms) 
                # a_hot[a] = 1 
                # x = np.kron(a_hot, x) 
            else: 
                x = self.bandit.features[self.iteration, a]
            self.phi[a] = x 

    def update_theta_hat(self): 
        for a in self.bandit.arms: 
            self.theta_hat[a] = self.A_inv[a] @ self.y_hat[a]

    def update_A_inv(self):
        self.A_inv[self.offline_action] = inv_sherman_morrison(
            self.phi[self.offline_action],
            self.A_inv[self.offline_action]
        )

    def update_y_hat(self): 
        self.y_hat[self.offline_action,:] += self.phi[self.offline_action] * self.bandit.rewards[self.iteration, self.offline_action] 

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
                
                # Update the offline grad_out and A_inv

                start_time = time.time() 
                self.update_phi()
                self.update_A_inv()
                self.update_y_hat()
                self.update_theta_hat() # (na, d)
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

                    print('\n[LinLCB] t={}, subopt={}, % optimal arms={}, action select time={}'.format(t, self.subopts[-1][1], \
                        self.opt_arm_select_percent_stats[-1][1], self.action_selection_times[-1][1] ))

                # increment counter
                self.iteration += 1

                # log
                update_time = sum([ item[1] for item in self.update_times]) / len(self.update_times) if len(self.update_times) > 0 else 0 
                postfix['update time'] = '{}'.format(update_time)


                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)