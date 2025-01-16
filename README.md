# Blackjack DQN
Basic DQN implementation for playing blackjack in a custom gym environment 

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
The agent used to play blackjack is an enhanced version of DQN proposed by Fujimoto et al., 2018 (https://arxiv.org/pdf/1802.09477). This technique:
- Maintains two separate Q-networks
- Takes the minimum of their products to prevent overoptimistic value estimates
- Clips the difference between the two networks' predictions

This creates a more conservative learning process, helping prevent the agent from learning risky stratagies based on overestimated rewards.

### Training Process
The agent learns through interactions across episodes. The approach is as follows:
1. Observe the current game state
2. Select an epsilon greedy action
3. Recieve a reward and the next state
4. Store the experience in a replay buffer
5. Periodically update networks based on saved experience




