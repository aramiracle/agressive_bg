"""
Empirical Match Equity Table

Maintains and updates match equity (probability of winning the match) for each
score pair. Learned from actual self-play outcomes, not hard-coded.

KEY DESIGN PRINCIPLES:
  - equity_table[(my_score, opp_score)] = P(I win match from here), in [0, 1]
  - Table is NOT symmetric across the diagonal by construction — it IS
    symmetric in the sense that equity(a,b) + equity(b,a) == 1.0 (zero-sum).
  - update_from_match must be called ONCE PER PLAYER PERSPECTIVE, not with
    mixed (my, opp) and (opp, my) pairs in the same call.
  - compute_reward does NOT multiply by 2.0: equity diffs are already in
    [-1, 1] and the value head (tanh) can represent them directly.
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

        At equal scores equity = 0.5.  The sigmoid mapping ensures
        equity(a,b) + equity(b,a) = 1.0 exactly at init, which is the
        only hard constraint the table must satisfy throughout training.
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
                    # Raw advantage: positive = I'm closer to winning
                    raw_advantage = (points_opp_need - points_i_need) / \
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

        Args:
            scores_seen_my_perspective : list of (my_score, opp_score) tuples
                                         encountered during the match, from
                                         THIS player's point of view only.
            i_won : bool — True if this player won the match.

        IMPORTANT: Call this method TWICE per match — once for each player —
        with the correct i_won flag for each.  Do NOT mix both players'
        (my, opp) and (opp, my) pairs into the same call.
        """
        target_equity = 1.0 if i_won else 0.0

        for my_score, opp_score in scores_seen_my_perspective:
            key = (my_score, opp_score)
            if key not in self.equity_table:
                continue
            current = self.equity_table[key]
            self.equity_table[key] += self.learning_rate * (target_equity - current)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def compute_reward(self, my_score_before, opp_score_before,
                       my_score_after, opp_score_after):
        """
        Reward = change in match win probability, in [-1, 1].

        equity is in [0, 1], so the diff is already in [-1, 1].
        We do NOT scale by 2 — the value head (tanh) has range [-1, 1]
        and can represent this directly.  Scaling by 2 would push targets
        outside the representable range and permanently miscalibrate the
        value head, breaking the REINFORCE advantage signal.
        """
        equity_before = self.get_equity(my_score_before, opp_score_before)
        equity_after  = self.get_equity(my_score_after,  opp_score_after)
        return equity_after - equity_before   # in [-1, 1], no extra scaling

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

        # Symmetry check: print max violation of equity(a,b)+equity(b,a)=1
        max_violation = 0.0
        for my in range(target):
            for opp in range(target):
                s = self.get_equity(my, opp) + self.get_equity(opp, my)
                max_violation = max(max_violation, abs(s - 1.0))
        print(f"  Max zero-sum symmetry violation: {max_violation:.4f}")