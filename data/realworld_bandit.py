"""Functions to create bandit problems from datasets."""

# %% 
"""
adult, mushroom, and shuttle
data: https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits
"""
import numpy as np
import pandas as pd
import os 
from easydict import EasyDict as edict

# data_root = '/scratch/tnguy258/datasets/neuralbandit/'
data_root = '/scratch/tnguy258/datasets/offline-neuralbandit-2/'
# data_root = '/scratch/tnguy258/datasets/offline-neuralbandit-tmp/'

os.makedirs(data_root, exist_ok=True)

CLSDataset = { # dataset: [context_dim, n_arms] # only n_arms is used, context_dim is inferred from the data as it changes in different trials 
    'mushroom': [117, 2], 
    'stock': [21, 8], 
    'jester': [32,8], # *
    'statlog': [9,7], # * 
    'adult': [93, 14], # *
    'census': [383, 9], 
    'covertype': [54,7], # *, not so good for neural?
    'mnist': [784,10], 
    'fashionmnist': [784,10]
}

# Add offline actions 
class CLSBandit(object):
    def __init__(self, 
            trial, 
            T, 
            name, 
            behavior_pi = 'eps-greedy', 
            behavior_epsilon=0.1, 
            num_test = 1000 
            ): # Create or load on trial basis. If saved, be consistent with T
        """
        datasets: (n, d)
        opt_vals[0]: optimal expected values, (n,) 
        opt_vals[1]: optimal actions, (n,)
        """
        self.name = name 
        self.T = T 
        self.num_test = num_test 
        self.trial = trial 
        self.n_arms = CLSDataset[name][1]
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
                datasets = arr['arr_0']
                opt_vals = arr['arr_1']
                actions = arr['arr_2']
        else:
            print('Creating new data into {}'.format(fname))
            if name == 'mushroom':
                datasets, opt_vals = sample_mushroom_data('./data/uci/mushroom.data', num_contexts=T + num_test)
            elif name == 'stock':
                datasets, opt_vals = sample_stock_data('./data/uci/raw_stock_contexts', num_contexts=T + num_test)
            elif name == 'jester':
                datasets, opt_vals = sample_jester_data('./data/uci/jester.npy', num_contexts=T + num_test)
            elif name == 'statlog': 
                datasets, opt_vals = sample_statlog_data('./data/uci/shuttle.trn', num_contexts=T + num_test, remove_underrepresented=True)
            elif name == 'adult': 
                datasets, opt_vals = sample_adult_data('./data/uci/adult.full', num_contexts=T + num_test)
            elif name == 'census': 
                datasets, opt_vals = sample_census_data('./data/uci/USCensus1990.data.txt', num_contexts=T + num_test)
            elif name == 'covertype': 
                datasets, opt_vals = sample_covertype_data('./data/uci/covtype.data', num_contexts=T + num_test)
            elif name == 'mnist':
                datasets, opt_vals = sample_mnist_data(num_contexts=T + num_test)
            elif name == 'fashionmnist':
                datasets, opt_vals = sample_mnist_data(num_contexts=T + num_test)
            else: 
                raise NotImplementedError 
            
            # actions = sample_offline_policy(opt_vals[1], datasets.shape[0], self.n_arms, behavior_pi='eps-greedy', behavior_epsilon=behavior_epsilon)
            
            actions = sample_eps_greedy_offline_policy(opt_vals[1], datasets.shape[0], self.n_arms,behavior_epsilon=behavior_epsilon)
            np.savez(fname, datasets, opt_vals, actions)

        
            
        self.features = datasets[:,:-self.n_arms] #(T, context_dim)

        self.context_dim = self.features.shape[1] # * self.n_arms

        # normalize features 
        self.features = normalize_contexts(self.features)
        # self.features = self.features[:,:CLSDataset[name][0]]
        self.rewards = datasets[:,-self.n_arms:] #(T, n_arms)
        self.best_rewards = opt_vals[0] # (T,)
        self.best_arms = opt_vals[1] # (T,)

        self.offline_arms = actions 

        self.real_T = self.features.shape[0] # mushroom contains only ~8k < 10k data points 

        print(self.features.shape, self.rewards.shape)
    

    @property 
    def arms(self): 
        return range(self.n_arms)


def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


def sample_mushroom_data(file_name,
                         num_contexts,
                         shuffle_rows=True, 
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
    """Samples bandit game from Mushroom UCI Dataset.

    Args:
        file_name: Route of file containing the original Mushroom UCI dataset.
        num_contexts: Number of points to sample, i.e. (context, action rewards).
        r_noeat: Reward for not eating a mushroom.
        r_eat_safe: Reward for eating a non-poisonous mushroom.
        r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
        r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
        prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.

    Returns:
        dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
        opt_vals: Vector of expected optimal (reward, action) for each context.

    We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
    """

    # first two cols of df encode whether mushroom is edible or poisonous
    df = pd.read_csv(file_name, header=None) # 8124 x 23 
    print(df.shape)
    df = one_hot(df, df.columns) # 8124 x 119 
    print(df.shape)


    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True if df.shape[0] < num_contexts else False)
    # ind = np.arange(num_contexts)
    # np.random.shuffle(ind)

    contexts = df.iloc[ind, 2:]
    # contexts = df.iloc[:, 2:] 
    # print(contexts)
    # if shuffle_rows:
        # np.random.shuffle(contexts)
    # contexts = contexts[:num_contexts, :]

    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    random_poison = np.random.choice(
        [r_eat_poison_bad, r_eat_poison_good],
        p=[prob_poison_bad, 1 - prob_poison_bad],
        size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    # import pdb
    # pdb.set_trace()
    # print(eat_reward, eat_reward.shape)
    eat_reward = eat_reward.values.reshape((num_contexts, 1))

    # compute optimal expected reward and optimal actions
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
        r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

    if r_noeat > exp_eat_poison_reward:
        # actions: no eat = 0 ; eat = 1
        opt_actions = df.iloc[ind, 0]  # indicator of edible
    else:
        # should always eat (higher expected reward)
        opt_actions = np.ones((num_contexts, 1))

    opt_vals = (opt_exp_reward.values, opt_actions.values)

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals


def sample_stock_data(file_name, context_dim=21, num_actions=8, num_contexts=1000,
                      sigma=0.01, shuffle_rows=True):
    """Samples linear bandit game from stock prices dataset.

    Args:
        file_name: Route of file containing the stock prices dataset.
        context_dim: Context dimension (i.e. vector with the price of each stock).
        num_actions: Number of actions (different linear portfolio strategies).
        num_contexts: Number of contexts to sample.
        sigma: Vector with additive noise levels for each action.
        shuffle_rows: If True, rows from original dataset are shuffled.

    Returns:
        dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).
        opt_vals: Vector of expected optimal (reward, action) for each context.
    """

    with open(file_name, 'r') as f:
        contexts = np.loadtxt(f, skiprows=1)

    print(contexts.shape)

    # contexts.shape[0] = 3713
    # if contexts.shape[0] < num_contexts - 1 :
    #     ind = np.random.choice(range(contexts.shape[0]), num_contexts, replace=True)
    #     contexts = contexts[ind]

    ind = np.random.choice(range(contexts.shape[0]), num_contexts, replace=True if contexts.shape[0] < num_contexts else False)
    contexts = contexts[ind]

    if shuffle_rows:
        np.random.shuffle(contexts)
    contexts = contexts[:num_contexts, :]

    betas = np.random.uniform(-1, 1, (context_dim, num_actions))
    betas /= np.linalg.norm(betas, axis=0)

    mean_rewards = np.dot(contexts, betas)
    noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
    rewards = mean_rewards + noise

    opt_actions = np.argmax(mean_rewards, axis=1).astype('int')
    opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
    return np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions)


def sample_jester_data(file_name, num_contexts, context_dim=32, num_actions=8,
                       shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.

    Args:
        file_name: Route of file containing the modified Jester dataset.
        context_dim: Context dimension (i.e. vector with some ratings from a user).
        num_actions: Number of actions (number of joke ratings to predict).
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        shuffle_cols: Whether or not context/action jokes are randomly shuffled.

    Returns:
        dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """

    with open(file_name, 'rb') as f:
        dataset = np.load(f)

    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]

    # if dataset.shape[0] < num_contexts - 1 :
    #     ind = np.random.choice(range(dataset.shape[0]), num_contexts, replace=True)
    #     dataset = dataset[ind]

    ind = np.random.choice(range(dataset.shape[0]), num_contexts, replace=True if dataset.shape[0] < num_contexts else False)
    dataset = dataset[ind]

    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a]
                            for i, a in enumerate(opt_actions)])

    return dataset, (opt_rewards, opt_actions)


def sample_statlog_data(file_name, num_contexts, shuffle_rows=True,
                        remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI statlog data.

    Args:
        file_name: Route of file containing the Statlog dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.

    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.

    https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
    """

    with open(file_name, 'r') as f:
        data = np.loadtxt(f)

    num_actions = 7  # some of the actions are very rarely optimal.

    # Shuffle data
    if shuffle_rows:
        np.random.shuffle(data)
    # data = data[:num_contexts, :]

    # if data.shape[0] < num_contexts - 1 :
    #     ind = np.random.choice(range(data.shape[0]), num_contexts, replace=True)
    #     data = data[ind]

    ind = np.random.choice(range(data.shape[0]), num_contexts, replace=True if data.shape[0] < num_contexts else False)
    data = data[ind]

    # Last column is label, rest are features
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int) - 1  # convert to 0 based index

    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)

    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_adult_data(file_name, num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI adult data.

    Args:
        file_name: Route of file containing the Adult dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.

    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.

    Preprocessing:
        * drop rows with missing values
        * convert categorical variables to 1 hot encoding

    https://archive.ics.uci.edu/ml/datasets/census+income
    """
    with open(file_name, 'r') as f:
        df = pd.read_csv(f, header=None,
                        na_values=[' ?']).dropna()

    num_actions = 14

    if shuffle_rows:
        df = df.sample(frac=1)


    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True if df.shape[0] < num_contexts else False)
    df = df.iloc[ind, :]

    labels = df[6].astype('category').cat.codes.to_numpy()
    df = df.drop([6], axis=1)

    # Convert categorical variables to 1 hot encoding
    cols_to_transform = [1, 3, 5, 7, 8, 9, 13, 14]
    df = pd.get_dummies(df, columns=cols_to_transform)

    if remove_underrepresented:
        df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.to_numpy()

    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_census_data(file_name, num_contexts, shuffle_rows=True,
                       remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI census data.

    Args:
        file_name: Route of file containing the Census dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.

    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.

    Preprocessing:
        * drop rows with missing labels
        * convert categorical variables to 1 hot encoding

    Note: this is the processed (not the 'raw') dataset. It contains a subset
    of the raw features and they've all been discretized.

    https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
    """
    # Note: this dataset is quite large. It will be slow to load and preprocess.
    with open(file_name, 'r') as f:
        df = (pd.read_csv(f, header=0, na_values=['?'])
            .dropna())

    num_actions = 9

    if shuffle_rows:
        df = df.sample(frac=1)

    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True if df.shape[0] < num_contexts else False)
    df = df.iloc[ind, :]

    # Assuming what the paper calls response variable is the label?
    labels = df['dOccup'].astype('category').cat.codes.to_numpy()
    # In addition to label, also drop the (unique?) key.
    df = df.drop(['dOccup', 'caseid'], axis=1)

    # All columns are categorical. Convert to 1 hot encoding.
    df = pd.get_dummies(df, columns=df.columns)

    if remove_underrepresented:
        df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.to_numpy()

    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_covertype_data(file_name, num_contexts, shuffle_rows=True,
                          remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI Cover_Type data.

    Args:
        file_name: Route of file containing the Covertype dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.

    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.

    Preprocessing:
        * drop rows with missing labels
        * convert categorical variables to 1 hot encoding

    https://archive.ics.uci.edu/ml/datasets/Covertype
    """
    with open(file_name, 'r') as f:
        df = (pd.read_csv(f, header=0, na_values=['?'])
            .dropna())

    num_actions = 7

    if shuffle_rows:
        df = df.sample(frac=1)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True if df.shape[0] < num_contexts else False)
    df = df.iloc[ind, :]
    # df = df.iloc[:num_contexts, :]

    # Assuming what the paper calls response variable is the label?
    # Last column is label.
    labels = df[df.columns[-1]].astype('category').cat.codes.to_numpy()
    df = df.drop([df.columns[-1]], axis=1)

    # All columns are either quantitative or already converted to 1 hot.
    if remove_underrepresented:
        df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.to_numpy()

    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_mnist_data(num_contexts): 
    from torchvision.datasets import MNIST
    from torchvision import transforms
    context_dim = 784
    dataset = MNIST('data/',train=True,transform=transforms.ToTensor(),download=True)
    contexts = dataset.data.view([-1,context_dim]).numpy()
    labels = dataset.targets.numpy()
    return classification_to_bandit_problem(contexts, labels, 10)

def sample_fashionmnist_data(num_contexts): 
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms
    context_dim = 784
    dataset = FashionMNIST('data/',train=True,transform=transforms.ToTensor(),download=True)
    contexts = dataset.data.view([-1,context_dim]).numpy()
    labels = dataset.targets.numpy()
    return classification_to_bandit_problem(contexts, labels, 10)

def classification_to_bandit_problem(contexts, labels, num_actions=None):
    """Normalize contexts and encode deterministic rewards."""

    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

    # Normalize features
    contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

    # normalize features so that their l2-norm is one

    # One hot encode labels as rewards
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0

    datasets = np.concatenate((contexts, rewards), axis=1) # (n, d + na)
    

    return datasets, (np.ones(contexts.shape[0]), labels)


def safe_std(values):
    """Remove zero std values for ones."""
    return np.array([val if val != 0.0 else 1.0 for val in values])


def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Removes classes when number of datapoints fraction is below a threshold."""

    # Threshold doesn't seem to agree with https://arxiv.org/pdf/1706.04687.pdf
    # Example: for Covertype, they report 4 classes after filtering, we get 7?
    total_count = labels.shape[0]
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts.astype('float') / total_count
    vals_and_ratios = dict(zip(unique, ratios))
    print('Unique classes and their ratio of total: %s' % vals_and_ratios)
    keep = [vals_and_ratios[v] >= thresh for v in labels]
    return features[keep], labels[np.array(keep)]

def normalize_contexts(contexts):
    """Normalize features such that its l2 norm is 1 
    args:
        contexts: (n,d)
    """
    return contexts / np.linalg.norm(contexts, ord=2, axis=1)[:,None] 

def sample_eps_greedy_offline_policy(opt_arms, n_contexts, n_arms, behavior_epsilon=0.1):
    """With probability eps, sample randomly any arms EXCEPT the optimal arm
    With probability 1 - eps, take the optimal arm
    """
    unif_arms = []
    for i in range(n_contexts):
        while True: 
            a = np.random.choice(n_arms) 
            if a != opt_arms[i]:
                break 
        unif_arms.append(a) 

    unif_arms = np.array(unif_arms)

    delta = np.random.uniform(size=(n_contexts,))
    selector = np.array(delta <= behavior_epsilon).astype('float32') 
    actions = selector.ravel() * unif_arms + (1 - selector.ravel()) * opt_arms
    actions = actions.astype('int')
    return actions



def sample_offline_policy(opt_mean_arms, num_contexts, num_actions, behavior_pi='eps-greedy', behavior_epsilon=0.1, subset_r = 0.5, 
                contexts=None, rewards=None): 
    """Sample offline actions 
    Args:
        opt_mean_arms: (num_contexts,)
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
        actions = selector.ravel() * uniform_actions + (1 - selector.ravel()) * opt_mean_arms 
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


# mushroom_ds = Mushroom(n_trials=30, T=10000)
# dataset, opt_vals = sample_stock_data('uci/raw_stock_contexts')

# %%
# mushroom_ds = Mushroom(n_trials=30, T=10000)
# stock_ds = Stock(n_trials=30, T=10000)
# jester_ds = Jester(n_trials=30, T=10000)
# statlog_ds = Statlog(n_trials=30, T=10000)
# adult_ds = Adult(n_trials=30, T=10000)
# census_ds = Census(n_trials=30, T=10000)

# T = 10000 
# # for name in CLSDataset: 
# for trial in range(30):
#     CLSBandit(trial, T, 'covertype')

# dataset, opt_vals = sample_mushroom_data('uci/mushroom.data', num_contexts=2) 

# clsbandit = CLSBandit(18, 10000, 'mushroom', behavior_epsilon=0.5)
# print(clsbandit.offline_arms[:20], sum(clsbandit.offline_arms[:20]) / 20)
# print(clsbandit.best_arms[:20])