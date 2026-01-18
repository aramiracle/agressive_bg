"""Compatibility shim - imports from new location."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backgammon.mcts import Node, MCTS

__all__ = ['Node', 'MCTS']
