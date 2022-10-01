"""Functions to create bandit problems from datasets."""

# %% 
"""
adult, mushroom, and shuttle
data: https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits

The text context is different from the train context as otherwise NeuralGreedy works too well 
"""
import numpy as np
import pandas as pd
import os 
from easydict import EasyDict as edict

# data_root = '/scratch/tnguy258/datasets/neuralbandit/'
data_root = '/scratch/tnguy258/datasets/offline-neuralbandit-2/' 
# test_data_root = '/scratch/tnguy258/datasets/offline-neuralbandit-test/'

# data_root = '/scratch/tnguy258/datasets/offline-neuralbandit-tmp/'

os.makedirs(data_root, exist_ok=True)
# os.makedirs(test_data_root, exist_ok=True)


# Add offline actions 
class SyntheticBandit(object):
    def __init__(self, 
            trial, 
            T, 
            context_dim, 
            num_actions, 
            noise_std, 
            name, 
            behavior_pi = 'eps-greedy', 
            behavior_epsilon=0.1,
            num_test = 1000 
            ): # Create or load on trial basis. If saved, be consistent with T

        self.num_test = num_test 
        self.name = name 
        self.T = T 
        self.trial = trial 
        self.n_arms = num_actions
        self.behavior_epsilon = behavior_epsilon # epsilon-greedy with respect to the optimal actions 
        self.behavior_pi = behavior_pi # behavior policy method 
        if self.behavior_pi == 'eps-greedy':
            pi_name = '{}-greedy'.format(behavior_epsilon) 
        else:
            raise NotImplementedError

        # self.context_dim = CLSDataset[name][0] * self.n_arms
        self.data_dir = os.path.join(data_root, name , pi_name)

        os.makedirs(self.data_dir, exist_ok=True)
        fname = os.path.join(self.data_dir, 'trial={}.npz'.format(trial))
        if os.path.exists(fname): 
                print('Loading data from {}'.format(fname))
                arr = np.load(fname)
                contexts = arr['arr_0']
                rewards = arr['arr_1']
                opt_rewards = arr['arr_2']
                opt_actions = arr['arr_3']
                off_actions = arr['arr_4']
        else:
            print('Creating new data into {}'.format(fname))
            if name == 'quadratic':
                contexts, rewards, opt_rewards, opt_actions = sample_quadratic(T + self.num_test, context_dim, num_actions, noise_std)
            elif name == 'quadratic2':
                contexts, rewards, opt_rewards, opt_actions = sample_quadratic2(T + self.num_test, context_dim, num_actions, noise_std)
            elif name == 'cosine':
                contexts, rewards, opt_rewards, opt_actions = sample_cosine(T + self.num_test, context_dim, num_actions, noise_std)
            elif name == 'exp':
                contexts, rewards, opt_rewards, opt_actions = sample_exp(T + self.num_test, context_dim, num_actions, noise_std)
            else: 
                raise NotImplementedError 
            
            off_actions = sample_offline_policy(opt_actions, contexts.shape[0], self.n_arms, behavior_pi='eps-greedy', behavior_epsilon=behavior_epsilon)
            np.savez(fname, contexts, rewards, opt_rewards, opt_actions, off_actions)

        self.features = contexts #(T, na, context_dim)
        self.context_dim = context_dim 
        self.rewards = rewards #(T, n_arms)
        self.best_rewards = opt_rewards # (T,)
        self.best_arms = opt_actions # (T,)
        self.offline_arms = off_actions 
        self.real_T = self.features.shape[0] # mushroom contains only ~8k < 10k data points 
        print(self.features.shape, self.rewards.shape)
    
    @property 
    def arms(self): 
        return range(self.n_arms)

def sample_quadratic(num_contexts, context_dim, num_actions, noise_std):
    thetas = np.random.uniform(-1,1,size = (context_dim, num_actions)) 
    thetas /= np.linalg.norm(thetas, axis=0)[None, :] # (d,a)
    h = lambda x: 10 * np.square(x @ thetas) # x: (n, d), h: (n, a) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim)) # (n,d)
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]
    mean_rewards = h(contexts) # (n, a)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions

def sample_quadratic2(num_contexts, context_dim, num_actions, noise_std):
    A = np.random.randn(context_dim, context_dim, num_actions) # (d,d,a)
    B = np.zeros((context_dim, context_dim, num_actions)) # (d,d,a)
    for a in range(num_actions):
        B[:,:,a] = A[:,:,a].T @ A[:,:,a] 
    h = lambda x: np.sum( np.dot(x,B) * x[:,:,None], axis=1) # x: (n, d), h: (n, a) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim))
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]

    mean_rewards = h(contexts) # (n, a)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions

def sample_cosine(num_contexts, context_dim, num_actions, noise_std):
    thetas = np.random.uniform(-1,1,size = (context_dim, num_actions)) 
    thetas /= np.linalg.norm(thetas, axis=0)[None, :] # (d,a)
    h = lambda x: np.cos(3 * x @ thetas) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim))
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]
    mean_rewards = h(contexts) # (T, na)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions

def sample_exp(num_contexts, context_dim, num_actions, noise_std):
    thetas = np.random.uniform(-1,1,size = (context_dim, num_actions)) 
    thetas /= np.linalg.norm(thetas, axis=0)[None, :] # (d,a)
    h = lambda x: np.exp(-10 * (x @ thetas)**2 ) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim))
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]
    mean_rewards = h(contexts) # (T, na)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions

def sample_offline_policy(opt_mean_rewards, num_contexts, num_actions, behavior_pi='eps-greedy', behavior_epsilon=0.1, subset_r = 0.5, 
                contexts=None, rewards=None): 
    """Sample offline actions 
    Args:
        opt_mean_rewards: (num_contexts,)
        num_contexts: int 
        num_actions: int
        pi: ['eps-greedy', 'subset', 'online']
    """
    # if pi == 'subset': # take only a subset of the action space? 
    #     subset_s = int(num_actions * subset_r)
    #     subset_mean_rewards = mean_rewards[np.arange(num_contexts), :subset_s]
    #     actions = np.argmax(subset_mean_rewards, axis=1)
    #     return actions 

    if behavior_pi == 'eps-greedy':
        uniform_actions = np.random.randint(low=0, high=num_actions, size=(num_contexts,)) 
        # opt_actions = np.argmax(mean_rewards, axis=1)
        delta = np.random.uniform(size=(num_contexts,))
        selector = np.array(delta <= behavior_epsilon).astype('float32') 
        actions = selector.ravel() * uniform_actions + (1 - selector.ravel()) * opt_mean_rewards 
        actions = actions.astype('int')
        return actions

    # elif pi == 'online':
    #     # Create offline data that is dependent on the past data
    #     assert contexts is not None 
    #     assert rewards is not None
    #     hparams = edict({
    #         'context_dim': contexts.shape[1], 
    #         'num_actions': num_actions, 
    #         'beta': 0.1, 
    #         'lambd0': 0.1, 
    #     })

    #     opt_actions = np.argmax(mean_rewards, axis=1)
    #     delta = np.random.uniform(size=(num_contexts,))
    #     selector = np.array(delta <= eps).astype('float32') 

    #     algo = LinUCB(hparams) # @TODO: To implement 

    #     algo.reset(1111)
    #     actions = []
    #     for i in tqdm(range(num_contexts),ncols=75):
    #         c = contexts[i:i+1,:]
    #         a_onl = algo.sample_action(c)
    #         # Combine a_onl and a_opt to make sure the offline data has a good coverage of the optimal policy
    #         a = selector[i] * a_onl + (1-selector[i]) * opt_actions[i:i+1]
    #         a = a.astype('int')
    #         r = rewards[i:i+1,a[0]:a[0]+1]  
    #         algo.update(c,a,r)
    #         actions.append(a[0])
    #     return np.array(actions).astype('int')
    # else:
    #     raise NotImplementedError('{} is not implemented'.format(pi))

# %%

# if __name__ == '__main__':
# data = 'stock' 
# num_contexts = 1000 
# if data == 'mushroom':
#     dataset, opt_vals = sample_mushroom_data('uci/mushroom.data', num_contexts=num_contexts) 
# elif data == 'stock':
#     dataset, opt_vals = sample_stock_data('uci/raw_stock_contexts', num_contexts=num_contexts) 

