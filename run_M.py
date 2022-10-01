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
parser.add_argument('--data',  type=str, default='quadratic') 
parser.add_argument('--algo',  type=str, default='NeuralPER')
parser.add_argument('--trials', nargs='+', type=int, default=['0'], help='list of trial indices')
parser.add_argument('--task', type=str, default='create_and_run_commands', choices=['create_and_run_commands'])
parser.add_argument('--behavior_epsilon', type=float, default=0.5)
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--perturbed_variance', type=float, default=1)
parser.add_argument('--reg_factor', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500, help='training epochs') # 500 for synthetic


parser.add_argument('--M_list', nargs='+', type=int, default=[1,2,5,10,20,30])
args = parser.parse_args()



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
        for M in args.M_list: 
            commands.append('python main_all.py --data {} --algo {} --learning_rate {} --reg_factor {} --perturbed_variance {} --M {} --trial {} --no-hpo --T 1000 --result_dir test-M-results/M={} --epochs 500'.format(args.data, \
                args.algo, args.learning_rate, args.reg_factor, args.perturbed_variance, M, trial, M))

    print(commands)

    if len(commands) < args.max_models_per_gpu: 
        args.models_per_gpu = len(commands)
    else: 
        args.models_per_gpu = args.max_models_per_gpu
    multi_gpu_launcher(commands, args.gpus, args.models_per_gpu)

if __name__ == '__main__':
    eval(args.task)()
