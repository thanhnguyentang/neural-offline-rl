# Neural Offline RL 
This is the official code of our *top-25%-noble* paper at ICLR'23: ["Provably Efficient Neural Offline Reinforcement Learning via Perturbed Rewards"](https://openreview.net/forum?id=WOquZTLCBO1). 

## Dependencies 
- torch 
- numpy 
- JAX 

## Instruction
- `main_all.py`: Run a specified algorithm in a specified dataset with specified hyperparameters. 
- `tune_all.py`: Tune the hyperparameters of each algorithm. 
- `run_all.py`: Run a tuned algorithm. 
- `nb/`: it contains notebooks to plot results. 
- `algorithms/`: all actual implementation of algorithms. 
- `data/`: constructs bandit and offline datasets. 
- `results/`: results of runing an experiment. 

## Bibliography

```
@inproceedings{nguyen-tang2023provably,
title={Provably Efficient Neural Offline Reinforcement Learning via Perturbed Rewards},
author={Thanh Nguyen-Tang and Raman Arora},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=WOquZTLCBO1}
}
```
