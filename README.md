# Blackjack DQN Agent
An implementation of Clipped Double Q-learning proposed by ([Fujimoto et al., 2018]([https://arxiv.org/pdf/1802.09477](https://arxiv.org/pdf/1802.09477))), that masters optimal Blackjack strategy through self-play. Built with a custom gym environment.

## Run the Code
1. Install the dependencies
```bash
pip install -r requirements.txt
```

2. Adjust the hyper parameters in `config.py` as desired. These include:
  - `learning_rate`: Step size for network updates (default: `1e-4`)
  - `buffer_size`: Maximum number of experiences to store (default: 25,000)
  - `batch_size`: Number of experiences to sample each update (default: 256)
  - `gamma`: Discount factor for future rewards (default: 0.99)
  - `target_update`: Number of steps between target network updates (default: 1000)
  - `epsilon_start`: Initial exploration rate (default: 1.0)
  - `epsilon_final`: Final exploration rate (default: 0.01)
  - `epsilon_decay`: Rate at which exploration decreases (default: 0.2)
  - `total_timesteps`: Total number of training steps (default: 100,000)
  - `eval_freq`: Number of steps between performance evaluations (default: 10,000)
  - `gradient_clip`: Max gradient value to avoid exploding gradients (default: 1.0)
  - `device`: Optional assignment to GPU (default: `cpu`)
  
  Note: The default values are tuned for optimal performance, but feel free to experiment.

3. Train the agent. The final model will be saved for use in analysis in the next step.
```bash
python driver.py train
```

Note: The repo includes an instance of an agent, `blackjack_agent.pth`, that is pretrained. Retraining will overwrite this agent at the path it is saved at. Further, `evaluate()` (discussed in detail below), assumes the saved agent path, but this can be manually changed in `driver.py`.

4. Evaluate performance. This will run 10,000 games and output performance statistics. 
```bash
python driver.py evaluate
```

Sample output:
```bash
=== Performance Summary ===
Wins: 4286 (42.9%)
Losses: 4858 (48.6%)
Draws: 856 (8.6%)
Average Reward: -0.057

=== Decision Analysis ===
Player Total | Hit Rate | Stand Rate | Total Decisions
-------------------------------------------------------
    12      |   93.1  % |    6.9   % |      1177      
    13      |   81.5  % |   18.5   % |      1327      
    14      |   59.7  % |   40.3   % |      1296      
    15      |   43.8  % |   56.2   % |      1332      
    16      |   29.8  % |   70.2   % |      1288      
    17      |   16.2  % |   83.8   % |      1305      
    18      |   10.5  % |   89.5   % |      1210      
    19      |   2.6   % |   97.4   % |      1179      
    20      |   0.0   % |   100.0  % |      1654      
```

Believe it or not, these results actually indicate viable learning of the basic optimal policy of blackjack- but more on that in the theory section.

## Theory

### 1. The Game of Blackjack

Blackjack serves as an excellent game candidate for the DQN agent. It has a discrete state space over a relatively small number of steps per episode. The rules of the game allow for the development of a basic optimal strategy, which is well documented.

The implementation used in this repository is a simplified version of the game:
- Removed 'splits' (separating pairs into separate games)
- Removed betting mechanism
- Focus on pure strategy optimization

#### A Note on House Edge
Blackjack is inherently a casino game where the rules favor the house:
- Dealer win rate: ~49%
- Player win rate: ~42%
- Remaining games: Draws

In a practical casino setting, an agent would need to incorporate a betting strategy to overcome the house edge - betting more aggressively on favorable hands to compensate for the inherent disadvantage. However, this project focuses purely on learning optimal play decisions.

Therefore, an appropriate success metric is achieving the theoretical optimal win rate of ~42% over a statistically significant number of games, demonstrating that the agent has learned the best possible strategy within the game's constraints.

### 2. Clipped Double Q-learning
The agent used to play blackjack is an enhanced version of DQN proposed by Fujimoto et al., 2018. This technique:
- Maintains two separate Q-networks
- Takes the minimum of their predictions to prevent overoptimistic value estimates
- Clips the difference between the two networks' predictions

This creates a more conservative learning process, helping prevent the agent from learning risky strategies based on overestimated rewards.

### Training Process
The agent learns through interactions across episodes. The approach is as follows:
1. Observe the current game state
2. Select an epsilon greedy action
3. Receive a reward and the next state
4. Store the experience in a replay buffer
5. Periodically update networks based on saved experience




