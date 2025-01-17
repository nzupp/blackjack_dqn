# -*- coding: utf-8 -*-
"""
Hyperparams for dqn_blackjack.py learning scenario. These are hand picked but
lead to viable optimal basic strategy.

@author: nzupp
"""
import torch

class Config:
    learning_rate = 1e-4
    buffer_size = 25000
    batch_size = 256
    gamma = 0.99
    target_update = 1000
    
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 0.2
    
    total_timesteps = 100000
    eval_freq = 10000
    
    gradient_clip = 1.0
    
    # Optional: cuda
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    