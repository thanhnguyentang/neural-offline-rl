"""This script runs one single algorithm in one single dataset at a single trial. 
"""
from typing import Text
import numpy as np 
import os 
import argparse
import random 
import torch 

from algorithms.neuraLCB import NeuraLCB
from algorithms.neuraLCBDiag import NeuraLCBDiag
from algorithms.neuralPER import NeuralPER
from algorithms.linLCB import LinLCB 
from algorithms.linPER import LinPER 

from algorithms.neuralGreedy import NeuralGreedy
from algorithms.utils import file_is_empty

from data.synthetic_bandit import SyntheticBandit
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # JAX pre-allocate 90% of acquired GPU mem by default, this line disables this feature.

parser = argparse.ArgumentParser()


# config = flags.FLAGS 

# Bandit 
parser.add_argument('--data', type=str, default='quadratic', help='dataset')
parser.add_argument('--algo', type=str, default='NeuraLCB', help='Algorithm to run.')


# Experiment 
parser.add_argument('--T', type=int, default=1000, help='Number of rounds')
parser.add_argument('--num_test', type=int, default=1000, help='Number of test samples')
parser.add_argument('--max_T', type=int, default=2000, help='Number of rounds')
parser.add_argument('--trial', nargs='+', type=int, default=[0], help='Trial number')
parser.add_argument('--hpo', default=True, action=argparse.BooleanOptionalAction, help='If True, tune hyperparams; if False, run with the best hyperparam')


# Neural network 
parser.add_argument('--n_layers',  type=int, default=3, help='Number of layers') 
parser.add_argument('--dropout_p', type=float, default= 0.2, help='Dropout probability')
parser.add_argument('--hidden_size', nargs='+', type=int, default = [64], help='Hidden size')

# Train config 
parser.add_argument('--batch_size', type=int, default=64, help='batch size') 
# parser.add_argument('--start_train_after', type=int, default= 500, help='Start training only after that many iterations. Make sure it is larger batch size.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate') 
parser.add_argument('--epochs', type=int, default=500, help='training epochs') 
parser.add_argument('--use_cuda', default=True, action=argparse.BooleanOptionalAction) 
parser.add_argument('--train_every', type=int, default=100, help='Train periodically')
parser.add_argument('--reg_factor', type=float, default=0.01, help='regularization factor')

# Offline data 
parser.add_argument('--behavior_pi', type=str, default='eps-greedy') 
parser.add_argument('--behavior_epsilon', type=float, default=0.5)
parser.add_argument('--context_dim', type=int, default=16)
parser.add_argument('--num_actions', type=int, default=10)
parser.add_argument('--noise_std', type=float, default=0.01)

# Test 
parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate periodically')


# NeuraLCB 
parser.add_argument('--beta', type=float, default=4, help='Exploration parameter') 

# NeuralTS 
parser.add_argument('--scaled_variance',  type=float, default=4, help='Scaled variance') 

# NeuralPR 
parser.add_argument('--perturbed_variance', type=float, default=0.1, help='Perturbed variance')
parser.add_argument('--M', type=int, default=10, help='Number of bootstraps')

# NeuralBoot, NeuralREx
parser.add_argument('--add_noise', default=True, action=argparse.BooleanOptionalAction, help = 'Add Langevin noise duing SGD') 
parser.add_argument('--ld_noise_std', type=float, default=0.01,help= 'Langevin noise std')
parser.add_argument('--n_bootstraps', type=int, default=10, help='Number of bootstrapped samples') 
parser.add_argument('--sample_every', type=int, default=100, help='Take bootstrapped samples periodically')
parser.add_argument('--gamma', type=float, default=0.1, help = 'Weighting bootstrapped sum') 
parser.add_argument('--warming_steps', type=int, default = 100, help = 'Warming steps before taking Langevin samples') 
parser.add_argument('--bootstrap_fractional', nargs='+', default=[0.5,0.5], help='Lower and upper quantile to sample bootstraps. Note that no space btw items')


# NeuralEpsGreedy 
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon')


__allowed_algos__ = ['NeuraLCB', 'LinLCB', 'NeuralPER', 'NeuralGreedy', 'NeuraLCBDiag', 'LinPER']
name2ind = {__allowed_algos__[i]:i for i in range(len(__allowed_algos__))}

firstline = {
    'NeuraLCB': 'lr | lambda | beta | opt-rate | regret\n', 
    'NeuraLCBDiag': 'lr | lambda | beta | opt-rate | regret\n', 
    'LinLCB': 'lr | lambda | beta | opt-rate | regret\n', 
    'LinPER': 'lr | lambda | perturbed-variance | M | opt-rate | regret\n', 
    'NeuralPER': 'lr | lambda | perturbed-variance | M | opt-rate | regret\n', 
    'NeuralGreedy': 'lr | lambda | regret\n'
}

config = parser.parse_args()

def set_randomness(seed): 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():    

    for trial in config.trial: 
        set_randomness(trial)
        bandit = SyntheticBandit(
            trial=trial, 
            T=config.max_T, 
            num_test=config.num_test, 
            context_dim=config.context_dim, 
            num_actions=config.num_actions, 
            noise_std=config.noise_std, 
            name=config.data, 
            behavior_pi = config.behavior_pi, 
            behavior_epsilon = config.behavior_epsilon)


        assert config.algo in __allowed_algos__ 

        print(config)

        for hidden_size in config.hidden_size: 

            result_dir = os.path.join('m-effect', 'mu_eps={}'.format(config.behavior_epsilon), config.data, 'm={}'.format(hidden_size), config.algo) 
            os.makedirs(result_dir, exist_ok=True)
            
            if config.algo == 'NeuraLCB': 
                algo = NeuraLCB(bandit,
                    # is_cls_bandit=False, 
                    T = config.T, 
                    hidden_size= hidden_size,
                    reg_factor= config.reg_factor,
                    n_layers= config.n_layers,
                    batch_size = config.batch_size,
                    p= config.dropout_p,
                    learning_rate= config.learning_rate,
                    epochs = config.epochs,
                    train_every = config.train_every,
                    evaluate_every = config.evaluate_every, 
                    use_cuda = config.use_cuda,
                    beta = config.beta)

            elif config.algo == 'NeuraLCBDiag': 
                algo = NeuraLCBDiag(bandit,
                    # is_cls_bandit=False, 
                    T = config.T, 
                    hidden_size= hidden_size,
                    reg_factor= config.reg_factor,
                    n_layers= config.n_layers,
                    batch_size = config.batch_size,
                    p= config.dropout_p,
                    learning_rate= config.learning_rate,
                    epochs = config.epochs,
                    train_every = config.train_every,
                    evaluate_every = config.evaluate_every,
                    use_cuda = config.use_cuda,
                    beta = config.beta)

            elif config.algo == 'LinLCB': 
                algo = LinLCB(bandit,
                    # is_cls_bandit=False,
                    T = config.T, 
                    reg_factor= config.reg_factor,
                    evaluate_every = config.evaluate_every,
                    beta = config.beta)

            elif config.algo == 'LinPER': 
                algo = LinPER(bandit,
                    # is_cls_bandit=False,
                    T = config.T, 
                    reg_factor= config.reg_factor,
                    M = config.M, 
                    evaluate_every = config.evaluate_every,
                    perturbed_variance = config.perturbed_variance)

            elif config.algo == 'NeuralPER':
                # NeuralPR 
                algo = NeuralPER(bandit,
                    T = config.T,
                    # is_cls_bandit=False, 
                    hidden_size=hidden_size,
                    reg_factor=config.reg_factor,
                    n_layers=config.n_layers,
                    batch_size = config.batch_size,
                    M = config.M, 
                    p=config.dropout_p,
                    learning_rate=config.learning_rate,
                    epochs=config.epochs,
                    train_every=config.train_every,
                    use_cuda=config.use_cuda,
                    evaluate_every = config.evaluate_every,
                    perturbed_variance = config.perturbed_variance)

            elif config.algo == 'NeuralGreedy': 
                # NeuralGreedy
                algo = NeuralGreedy(bandit,
                    T = config.T,
                    # is_cls_bandit=False, 
                    hidden_size=hidden_size,
                    reg_factor=config.reg_factor,
                    n_layers=config.n_layers,
                    batch_size = config.batch_size,
                    p=config.dropout_p,
                    learning_rate=config.learning_rate,
                    epochs=config.epochs,
                    train_every=config.train_every,
                    evaluate_every = config.evaluate_every,
                    use_cuda=config.use_cuda,
                    )

            else:
                raise NotImplementedError 
        
            algo.run() 
            config.hpo = False 

            if not config.hpo:
                print('Make sure you are using the best hyperparameters as it will overwrite the result file.')
                subopts = np.array(algo.subopts) # (t, sub-opt)
                update_time = np.array(algo.update_times) # (t, time)
                action_selection_time = np.array(algo.action_selection_times) # (t, time)
                fname = os.path.join(result_dir, 'trial={}.npz'.format(trial))
                np.savez(fname, subopts, update_time, action_selection_time)

if __name__ == '__main__': 
    # app.run(main)
    main()
