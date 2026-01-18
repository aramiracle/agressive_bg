#!/usr/bin/env python3
"""Entry point for training the backgammon AI."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from backgammon.trainer import train

if __name__ == "__main__":
    train()

