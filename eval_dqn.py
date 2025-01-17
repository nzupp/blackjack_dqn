# -*- coding: utf-8 -*-
"""
Let's see how well it does!

Note:
Like most casino style games, the edge is with the dealer. We can 
expect the dealer to win ~49% of the time, the player to win ~42% 
of the time, and a draw the remainder of the time

@author: nzupp
"""
from blackjack_env import BlackjackEnv
from dqn_blackjack import DQNAgent
from tqdm import tqdm

def analyze_performance(agent):
    env = BlackjackEnv()
    wins = losses = draws = blackjacks = total_reward = 0
    decisions = {i: {'hit': 0, 'stand': 0} for i in range(4, 22)}
    
    n_games=10000
    
    for _ in tqdm(range(n_games), desc='Analyzing Performance'):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, training=False)
            player_total = int(state[6])
            decisions[player_total]['hit' if action == 1 else 'stand'] += 1
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        if episode_reward > 0:
            wins += 1
            if episode_reward == 1.5:
                blackjacks += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
    
    win_rate = (wins / n_games) * 100
    blackjack_rate = (blackjacks / n_games) * 100
    avg_reward = total_reward / n_games
    
    print("\n=== Performance Summary ===")
    print(f"Wins: {wins} ({(wins/n_games)*100:.1f}%)")
    print(f"Losses: {losses} ({(losses/n_games)*100:.1f}%)")
    print(f"Draws: {draws} ({(draws/n_games)*100:.1f}%)")
    print(f"Average Reward: {avg_reward:.3f}")
    
    print("\n=== Decision Analysis ===")
    print("Player Total | Hit Rate | Stand Rate | Total Decisions")
    print("-" * 55)
    
    for total in range(12, 21):
        hit = decisions[total]['hit']
        stand = decisions[total]['stand']
        total_decisions = hit + stand
        if total_decisions > 0:
            hit_rate = (hit / total_decisions) * 100
            print(f"{total:^11} | {hit_rate:^8.1f}% | {100-hit_rate:^9.1f}% | {total_decisions:^15}")
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'blackjacks': blackjacks,
        'win_rate': win_rate,
        'blackjack_rate': blackjack_rate,
        'avg_reward': avg_reward,
        'decisions': decisions
    }

