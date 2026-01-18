#!/usr/bin/env python3
"""Entry point for the Kivy desktop UI."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import and run the Kivy app
from backgammon.ui_desktop import BackgammonApp

if __name__ == '__main__':
    BackgammonApp().run()

