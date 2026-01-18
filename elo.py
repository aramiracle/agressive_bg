"""Compatibility shim - imports from new location."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backgammon.elo import (
    calculate_expected_score,
    update_elo,
    play_single_game,
    evaluate_vs_opponent
)

__all__ = [
    'calculate_expected_score',
    'update_elo',
    'play_single_game',
    'evaluate_vs_opponent'
]
