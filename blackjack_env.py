# -*- coding: utf-8 -*-
"""
Custom Blackjack environment wrapped in gym for easy RL training.
This is a very limited implementation, with no betting strategy,
no splitting, and one deck shuffled each play through.

The game is state represented in eight dimensions, which could be considered
arbitrary, but leads to desireable overall performance in experiments

@author: nzupp
"""
import gym
from gym import spaces
import numpy as np

class BlackjackEnv(gym.Env):
    def __init__(self):
        super(BlackjackEnv, self).__init__()
        
        self.card_values = {
            'A': 11, 'K': 10, 'Q': 10, 'J': 10, '10': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
            '4': 4, '3': 3, '2': 2
        }
        
        self.cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.suits = ['H', 'D', 'C', 'S']
        
        # Action space: 0 = stay, 1 = hit
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 0, 0, 0, 1, 2, 0]),
            high=np.array([13, 13, 13, 13, 13, 13, 21, 1]),
            dtype=np.int32
        )
        
        self.card_to_int = {card: idx+1 for idx, card in enumerate(self.cards)}
        self.reset()

    # Blackjack scoring, inclduing handling Ace
    def _get_score(self, cards):
        score = 0
        ace_count = 0
        
        for card, _ in cards:
            if card == 'A':
                ace_count += 1
                score += 11
            else:
                score += self.card_values[card]
        
        while score > 21 and ace_count > 0:
            score -= 10
            ace_count -= 1
            
        return score, ace_count > 0

    # I represent the state in a simplified manner. There are 5 dimensions for
    # player cards, which could be experimented with. The dealer card is included
    # in the state. The current player score, and the number of aces is also
    # included.
    def _get_obs(self):
        player_cards = [self.card_to_int[card] for card, _ in self.player_cards]
        
        while len(player_cards) < 5:
            player_cards.append(0)
            
        dealer_card = self.card_to_int[self.dealer_cards[0][0]]
        player_score, has_usable_ace = self._get_score(self.player_cards)
        
        return np.array([
            player_cards[0],
            player_cards[1],
            player_cards[2],
            player_cards[3],
            player_cards[4],
            dealer_card,
            player_score,
            int(has_usable_ace)
        ])

    # Gym reset function that initializes the game state
    def reset(self):
        self.deck = [(card, suit) for card in self.cards for suit in self.suits]
        np.random.shuffle(self.deck)
        
        self.player_cards = [self.deck.pop(), self.deck.pop()]
        self.dealer_cards = [self.deck.pop(), self.deck.pop()]
        
        player_score, _ = self._get_score(self.player_cards)
        dealer_score, _ = self._get_score(self.dealer_cards)
        
        if player_score == 21:
            if dealer_score == 21:
                self.done = True
                self.reward = 0
            else:
                self.done = True
                self.reward = 1
        else:
            self.done = False
            self.reward = 0
            
        return self._get_obs()

    # Gym step function
    def step(self, action):
        assert self.action_space.contains(action)
        
        if self.done:
            return self._get_obs(), self.reward, self.done, {}
            
        # Player's turn
        if action == 1:  # hit
            self.player_cards.append(self.deck.pop())
            player_score, _ = self._get_score(self.player_cards)
            
            if player_score > 21:
                self.done = True
                self.reward = -1
                return self._get_obs(), self.reward, self.done, {}
                
        else:  # stay
            # Dealer's turn
            dealer_score, _ = self._get_score(self.dealer_cards)
            
            while dealer_score < 17:
                self.dealer_cards.append(self.deck.pop())
                dealer_score, _ = self._get_score(self.dealer_cards)
            
            player_score, _ = self._get_score(self.player_cards)
            
            if dealer_score > 21:
                self.reward = 1
            elif dealer_score > player_score:
                self.reward = -1
            elif dealer_score < player_score:
                self.reward = 1
            else:
                self.reward = 0
                
            self.done = True
            
        return self._get_obs(), self.reward, self.done, {}

    # Very basic render function
    def render(self, mode='human'):
        pass
