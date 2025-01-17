# -*- coding: utf-8 -*-
"""
Driver for training and evaluation

@author: nzupp
"""
from dqn_blackjack import DQNAgent, train_dqn
from eval_dqn import analyze_performance
import sys

def train():
    train_dqn()

def evaluate():
    try:
        agent = DQNAgent()
        agent.load('blackjack_agent.pth')
        analyze_performance(agent)
    except:
        print("error")

if __name__ == "__main__":
    command = sys.argv[1].lower()
    if command == "train":
        train()
    elif command == "evaluate":
        evaluate()
    else:
        print("Invalid command. Use 'train' or 'evaluate'")
