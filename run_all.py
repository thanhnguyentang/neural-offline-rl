"""Run bandit algorithms with the best hyperparameters on classification bandits. 
"""
import os
import subprocess
import time
import argparse
import random
import glob
import numpy as np
import errno


parser = argparse.ArgumentParser()
parser.add_argument('--models_per_gpu', type=int, default=30)
parser.add_argument('--max_models_per_gpu', type=int, default=30)
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='gpus indices used for multi_gpu')
parser.add_argument('--data', nargs='+', type=str, default=['quadratic']) 
parser.add_argument('--algo', nargs='+', type=str, default=['NeuraLCB'])
parser.add_argument('--trials', nargs='+', type=int, default=['0'], help='list of trial indices')
parser.add_argument('--task', type=str, default='create_and_run_commands', choices=['create_and_run_commands'])
parser.add_argument('--behavior_epsilon', type=float, default=0.5)
parser.add_argument('--T', type=int, default=1000)

args = parser.parse_args()

no_tune_algos = ['NeuralGreedy']
synthetic_group = ['quadratic', 'quadratic2', 'cosine', 'exp']


def extract_hyperparameters(algo, data):
    """Extract the best hyperparameters"""
    result_dir = os.path.join('results', 'mu_eps={}'.format(args.behavior_epsilon), data, algo) 
    fname = os.path.join(result_dir, 'tune.txt')
    try: 
        with open(fname, 'r') as fo: 
            lines = fo.readlines()

        lines = lines[1:] 
        regrets = []
        for i, line in enumerate(lines):
            regret = float(line.split('|')[-1])
            regrets.append(regret) 
        min_ind = np.argmin(np.array(regrets))
        best_line = lines[min_ind].split('|')
        return best_line

    except:
        print('Optimized hyperparameters not found. Consider run `tune_realworld.py` first')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)

def multi_gpu_launcher(commands,gpus,models_per_gpu):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    procs = [None]*len(gpus)*models_per_gpu

    while len(commands) > 0:
        for i,proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this index; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs[i] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()

def create_and_run_commands(): 
    commands = []
    for trial in args.trials:
        for data in args.data:
            # HAND-CRAFT T 
            if data in synthetic_group: 
                args.T = 5000 
            else:
                args.T = 10000
            for algo in args.algo:
                if algo not in no_tune_algos: # no tuning for NeuralGreedy
                    best_line = extract_hyperparameters(algo, data) 
                    print('Algo: {} Data: {} Best-hyperparams: {}'.format(algo, data, best_line))
                else: # use lr from NeuraLCB
                    try:
                        best_line = extract_hyperparameters('NeuraLCB', data) 
                    except: 
                        best_line = extract_hyperparameters('NeuraLCBDiag', data) 
                    print('Algo: {} Data: {} Best-hyperparams: {}'.format(algo, data, best_line[:1]))


                if algo in ['NeuraLCB', 'LinLCB', 'NeuraLCBDiag']:
                    commands.append('python main_all.py --data {} --algo {} --learning_rate {} --reg_factor {} --beta {} --trial {} --no-hpo --T {}'.format(data, algo, best_line[0], best_line[1], best_line[2], trial, args.T))
                    
                elif algo in ['NeuralPER', 'LinPER']: 
                    commands.append('python main_all.py --data {} --algo {} --learning_rate {} --reg_factor {} --perturbed_variance {} --M {} --trial {} --no-hpo --T {}'.format(data, algo, best_line[0], best_line[1], best_line[2], best_line[3], trial, args.T))

                elif algo in no_tune_algos: # no tuning for NeuralGreedy, use lr from NeuraLCB
                    commands.append('python main_all.py --data {} --algo {} --learning_rate {} --reg_factor {} --trial {} --no-hpo --T {}'.format(data, algo, best_line[0], best_line[1], trial, args.T))
    print(commands)

    if len(commands) < args.max_models_per_gpu: 
        args.models_per_gpu = len(commands)
    else: 
        args.models_per_gpu = args.max_models_per_gpu
    multi_gpu_launcher(commands, args.gpus, args.models_per_gpu)

if __name__ == '__main__':
    eval(args.task)()
    # best_line = extract_hyperparameters('NeuralUCB', 'mushroom')
    # print(best_line)