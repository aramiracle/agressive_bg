"""
Empirical Match Equity Table

Maintains and updates match equity (probability of winning the match) for each
score pair. This is learned from actual self-play outcomes, not hard-coded.

The match equity table solves the cube inflation problem: it captures the TRUE
strategic value of winning/losing points at different scores, independent of
how those points were obtained (cube level).
"""

import torch
import math
from src.config import Config


class MatchEquityTable:
    def __init__(self, match_target=None, learning_rate=0.01):
        """
        Initialize match equity table with symmetric starting values.
        
        Args:
            match_target: Points needed to win match (default: Config.MATCH_TARGET)
            learning_rate: How quickly to update equity estimates (default: 0.01)
        """
        self.match_target = match_target if match_target is not None else Config.MATCH_TARGET
        self.learning_rate = learning_rate
        
        # equity_table[(my_score, opp_score)] = probability I win the match from here
        # Values are in [0, 1]
        self.equity_table = {}
        
        self._initialize_table()
    
    def _initialize_table(self):
        """
        Initialize with symmetric, sensible starting values.
        
        Theory: At equal scores, equity = 50%. As I get ahead, equity increases
        smoothly. We use a sigmoid-like function for initialization.
        """
        target = self.match_target
        
        for my_score in range(target + 1):
            for opp_score in range(target + 1):
                if my_score >= target:
                    # I've won the match
                    equity = 1.0
                elif opp_score >= target:
                    # Opponent won the match
                    equity = 0.0
                else:
                    # Both still playing: estimate based on score difference
                    # This is a rough heuristic just for initialization
                    # The table will quickly learn the true values from experience
                    score_diff = my_score - opp_score
                    points_i_need = target - my_score
                    points_opp_needs = target - opp_score
                    
                    # Sigmoid-like mapping: more ahead = higher equity
                    # At equal scores (diff=0): equity ≈ 0.5
                    # Each point ahead increases equity by ~5-10%
                    raw_advantage = score_diff / (points_i_need + points_opp_needs + 1)
                    equity = 1.0 / (1.0 + math.exp(-3.0 * raw_advantage))
                
                self.equity_table[(my_score, opp_score)] = equity
    
    def get_equity(self, my_score, opp_score):
        """
        Get match equity for a given score.
        
        Returns probability of winning the match from this score (in [0, 1]).
        """
        key = (my_score, opp_score)
        if key not in self.equity_table:
            # Shouldn't happen if properly initialized, but handle gracefully
            return 0.5
        return self.equity_table[key]
    
    def update_from_match(self, scores_seen, winner):
        """
        Update equity table based on a completed match.
        
        Args:
            scores_seen: List of (my_score, opp_score) tuples encountered during match
            winner: Who won the match: +1 or -1
        
        Each score pair seen in the match gets updated toward 1.0 (if I won) or
        0.0 (if I lost). This is standard Monte Carlo update for state values.
        """
        target_equity = 1.0 if winner == 1 else 0.0
        
        for my_score, opp_score in scores_seen:
            key = (my_score, opp_score)
            if key not in self.equity_table:
                continue
            
            current_equity = self.equity_table[key]
            # Move toward target with learning rate (exponential moving average)
            self.equity_table[key] += self.learning_rate * (target_equity - current_equity)
    
    def compute_reward(self, my_score_before, opp_score_before, 
                       my_score_after, opp_score_after):
        """
        Compute reward as the CHANGE in match equity.
        
        This is the key function: it converts a game outcome (score change) into
        a training target for the value head.
        
        Args:
            my_score_before: My score before the game
            opp_score_before: Opponent score before the game
            my_score_after: My score after the game
            opp_score_after: Opponent score after the game
        
        Returns:
            Reward in [-1, 1], representing the change in match win probability
        """
        equity_before = self.get_equity(my_score_before, opp_score_before)
        equity_after = self.get_equity(my_score_after, opp_score_after)
        
        # Change in equity, scaled to [-1, 1]
        # equity is in [0, 1], so diff is in [-1, 1]
        # We scale by 2 to use the full [-1, 1] range for value head targets
        reward = (equity_after - equity_before) * 2.0
        
        return reward
    
    def save(self, filepath):
        """Save equity table to disk."""
        torch.save({
            'equity_table': self.equity_table,
            'match_target': self.match_target,
            'learning_rate': self.learning_rate
        }, filepath)
    
    def load(self, filepath):
        """Load equity table from disk."""
        checkpoint = torch.load(filepath)
        self.equity_table = checkpoint['equity_table']
        self.match_target = checkpoint['match_target']
        self.learning_rate = checkpoint['learning_rate']
    
    def print_table(self):
        """Print the equity table for debugging/inspection."""
        print(f"\nMatch Equity Table (match to {self.match_target}):")
        print("=" * 60)
        print("My\\Opp", end="")
        for opp in range(self.match_target + 1):
            print(f"{opp:6}", end="")
        print()
        print("-" * 60)
        
        for my in range(self.match_target + 1):
            print(f"{my:6}", end="")
            for opp in range(self.match_target + 1):
                equity = self.get_equity(my, opp)
                print(f"{equity:6.2f}", end="")
            print()
        print("=" * 60)