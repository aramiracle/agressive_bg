"""
Empirical Match Equity Table

Maintains and updates match equity (probability of winning the match) for each
score pair. Learned from actual self-play outcomes, not hard-coded.

KEY DESIGN PRINCIPLES:
  - equity_table[(my_score, opp_score)] = P(I win match from here), in [0, 1]
  - Zero-sum: equity(a,b) + equity(b,a) == 1.0
  - update_from_match must be called ONCE PER PLAYER PERSPECTIVE.

VALUE TARGET vs REWARD — critical distinction:
  - value_target: what the value head (tanh, range [-1,1]) trains on.
                  = 2 * equity_after - 1
                  Spans the full [-1,1] range, enabling meaningful
                  positional evaluation. This is compute_value_target().
  - equity_change: used ONLY for the ME soft target (cube JS loss) and
                   PER priorities.  Small magnitude (~±0.1 per game).
                   This is compute_equity_change().
  - compute_reward() is an alias for compute_value_target() to keep
    the existing call-site in game.py working without changes.
"""

import torch
import math
from src.config import Config


class MatchEquityTable:
    def __init__(self, match_target=None, learning_rate=0.01):
        self.match_target  = match_target if match_target is not None else Config.MATCH_TARGET
        self.learning_rate = learning_rate
        self.equity_table  = {}
        self._initialize_table()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_table(self):
        """
        Initialize with zero-sum symmetric values.
        equity(a,b) + equity(b,a) == 1.0 exactly at init.
        At equal scores equity == 0.5.
        """
        target = self.match_target

        for my_score in range(target + 1):
            for opp_score in range(target + 1):
                if my_score >= target:
                    equity = 1.0
                elif opp_score >= target:
                    equity = 0.0
                else:
                    points_i_need   = target - my_score
                    points_opp_need = target - opp_score
                    raw_advantage   = (points_opp_need - points_i_need) / \
                                      (points_i_need + points_opp_need)
                    equity = 1.0 / (1.0 + math.exp(-4.0 * raw_advantage))

                self.equity_table[(my_score, opp_score)] = equity

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get_equity(self, my_score, opp_score):
        key = (my_score, opp_score)
        if key not in self.equity_table:
            return 0.5
        return self.equity_table[key]

    # ------------------------------------------------------------------
    # Update — ONE PERSPECTIVE PER CALL
    # ------------------------------------------------------------------
    def update_from_match(self, scores_seen_my_perspective, i_won):
        """
        Update equity table for ONE player's perspective.

        Call TWICE per match — once per player — with the correct i_won flag.
        Do NOT mix both players' tuples in the same call.
        """
        target_equity = 1.0 if i_won else 0.0
        for my_score, opp_score in scores_seen_my_perspective:
            key = (my_score, opp_score)
            if key not in self.equity_table:
                continue
            current = self.equity_table[key]
            self.equity_table[key] += self.learning_rate * (target_equity - current)

    # ------------------------------------------------------------------
    # Value target — what the value head trains on
    # ------------------------------------------------------------------
    def compute_value_target(self, my_score_before, opp_score_before,
                             my_score_after, opp_score_after):
        """
        Returns the VALUE HEAD training target in [-1, 1].

        = 2 * equity_after - 1

        This maps equity [0, 1] to [-1, 1], matching the tanh output range
        and ensuring the value head receives targets that span its full
        representable range.  Without this mapping, equity changes (~±0.1)
        are too small for the value head to learn meaningful positional
        quality, collapsing all predictions toward zero.

        Example (7pt match, trailing 0-0 → 1-0 after winning):
            equity_after  = get_equity(1, 0) ≈ 0.60
            value_target  = 2 * 0.60 - 1    = 0.20   ← meaningful signal
            equity_change = 0.60 - 0.50      = 0.10   ← too small alone
        """
        equity_after = self.get_equity(my_score_after, opp_score_after)
        return 2.0 * equity_after - 1.0

    def compute_reward(self, my_score_before, opp_score_before,
                       my_score_after, opp_score_after):
        """Alias for compute_value_target — keeps existing call-sites working."""
        return self.compute_value_target(
            my_score_before, opp_score_before,
            my_score_after,  opp_score_after,
        )

    # ------------------------------------------------------------------
    # Equity change — used for ME soft target and PER priorities
    # ------------------------------------------------------------------
    def compute_equity_change(self, my_score_before, opp_score_before,
                              my_score_after, opp_score_after):
        """
        Returns the raw equity change in [-1, 1].
        Used by compute_me_soft_target() to build the cube JS target.
        Small magnitude (~±0.05–0.15 per game), NOT suitable as a
        standalone value head target.
        """
        equity_before = self.get_equity(my_score_before, opp_score_before)
        equity_after  = self.get_equity(my_score_after,  opp_score_after)
        return equity_after - equity_before

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath):
        torch.save({
            'equity_table':  self.equity_table,
            'match_target':  self.match_target,
            'learning_rate': self.learning_rate,
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.equity_table  = checkpoint['equity_table']
        self.match_target  = checkpoint['match_target']
        self.learning_rate = checkpoint['learning_rate']

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def print_table(self):
        target = self.match_target
        print(f"\nMatch Equity Table (match to {target}):")
        print("=" * 60)
        print(f"{'My/Opp':>6}", end="")
        for opp in range(target + 1):
            print(f"{opp:6}", end="")
        print()
        print("-" * 60)
        for my in range(target + 1):
            print(f"{my:6}", end="")
            for opp in range(target + 1):
                print(f"{self.get_equity(my, opp):6.2f}", end="")
            print()
        print("=" * 60)

        max_violation = 0.0
        for my in range(target):
            for opp in range(target):
                s = self.get_equity(my, opp) + self.get_equity(opp, my)
                max_violation = max(max_violation, abs(s - 1.0))
        print(f"  Max zero-sum symmetry violation: {max_violation:.4f}")