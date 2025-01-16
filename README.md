# Blackjack DQN
Basic DQN implementation for playing blackjack in a custom gym environment 

### 2. The Game of Blackjack

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
