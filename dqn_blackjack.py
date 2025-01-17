# -*- coding: utf-8 -*-
"""
Clipped Double Q-Learning implementation for Blackjack.

The agent learns to play Blackjack against a standard dealer, achieving 
performance close to optimal basic strategy (~42-43% win rate).

@author: nzupp
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from blackjack_env import BlackjackEnv
from tqdm import tqdm
from config import Config

# Basic Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Experience replay buffer to store and sample transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# Hyperparameters and class logic of the DQN Agent
# These have been tuned by hand at the moment
class DQNAgent:
    def __init__(self, config=None, state_dim=8, action_dim=2):
        if config is None:
            config = Config()
            
        self.config = config
        self.action_dim = action_dim
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.target_update = config.target_update
        self.device = config.device
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                  lr=config.learning_rate)
        self.memory = ReplayBuffer(config.buffer_size)
        
        self.epsilon_start = config.epsilon_start
        self.epsilon_final = config.epsilon_final
        self.epsilon_decay = config.epsilon_decay
        
        self.steps = 0
    
    # Epsilon greedy
    def get_epsilon(self, total_steps):
        fraction = min(float(self.steps) / (total_steps * self.epsilon_decay), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)
    
    # Epsilon decay steps currently hardcoded to same size of training steps
    def select_action(self, state, training=True):
        if training and random.random() < self.get_epsilon(50000):
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    # Core learning methodology:
    # - Sample a batch of experiences from memory
    # - Compute current Q-values from the policy network
    # - Target Q-Value Computation:
    #    - Use policy network to select best actions for next states
    #    - Use target network to evaluate those actions
    # - Update the policy network using backpropagation
    # - Periodically update the target network
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clippin
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    
    # Save and load utility functions
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

def evaluate_agent(agent, env, n_episodes=1000):
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_reward += episode_reward
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
    
    return {
        'win_rate': wins / n_episodes,
        'loss_rate': losses / n_episodes,
        'draw_rate': draws / n_episodes,
        'avg_reward': total_reward / n_episodes
    }

# The actual training loop, inlcuding train step, with total time steps set
def train_dqn(config=None):
    if config is None:
        config = Config()
        
    env = BlackjackEnv()
    agent = DQNAgent(config)
    
    metrics = {
        'win_rates': [],
        'loss_rates': [],
        'draw_rates': [],
        'rewards': [],
        'losses': [],
        'timesteps': []
    }
    
    state = env.reset()
    total_reward = 0
    episode_reward = 0
    
    pbar = tqdm(range(1, config.total_timesteps + 1), desc='Training')
    for step in pbar:
        agent.steps = step
        
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        agent.memory.push(state, action, reward, next_state, done)
        
        state = next_state
        
        loss = agent.train_step()
        if loss:
            metrics['losses'].append(loss)
        
        if done:
            total_reward += episode_reward
            episode_reward = 0
            state = env.reset()
        
        if step % config.eval_freq == 0:
            eval_metrics = evaluate_agent(agent, env)
            metrics['win_rates'].append(eval_metrics['win_rate'])
            metrics['loss_rates'].append(eval_metrics['loss_rate'])
            metrics['draw_rates'].append(eval_metrics['draw_rate'])
            metrics['rewards'].append(eval_metrics['avg_reward'])
            metrics['timesteps'].append(step)
            
            pbar.set_postfix({
                'win_rate': f"{eval_metrics['win_rate']:.3f}",
                'loss_rate': f"{eval_metrics['loss_rate']:.3f}",
                'avg_reward': f"{eval_metrics['avg_reward']:.3f}"
            })
    
    agent.save('blackjack_agent.pth')
