"""Compatibility shim - imports from new location."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backgammon.ui_desktop import (
    DiceWidget,
    BackgammonBoard,
    GameScreen,
    BackgammonApp,
    COLOR_FELT,
    COLOR_FELT_DARK,
    COLOR_WOOD_DARK,
    COLOR_WOOD_MID,
    COLOR_WOOD_LIGHT,
    COLOR_WOOD_FRAME,
    COLOR_POINT_LIGHT,
    COLOR_POINT_DARK,
    COLOR_WHITE_CHECKER,
    COLOR_BLACK_CHECKER,
    COLOR_HIGHLIGHT,
    COLOR_SELECTED,
    COLOR_DICE_BG,
    COLOR_DICE_DOT,
    COLOR_CUBE,
    COLOR_CUBE_TEXT,
)

__all__ = [
    'DiceWidget',
    'BackgammonBoard',
    'GameScreen',
    'BackgammonApp',
    'COLOR_FELT',
    'COLOR_FELT_DARK',
    'COLOR_WOOD_DARK',
    'COLOR_WOOD_MID',
    'COLOR_WOOD_LIGHT',
    'COLOR_WOOD_FRAME',
    'COLOR_POINT_LIGHT',
    'COLOR_POINT_DARK',
    'COLOR_WHITE_CHECKER',
    'COLOR_BLACK_CHECKER',
    'COLOR_HIGHLIGHT',
    'COLOR_SELECTED',
    'COLOR_DICE_BG',
    'COLOR_DICE_DOT',
    'COLOR_CUBE',
    'COLOR_CUBE_TEXT',
]

if __name__ == '__main__':
    BackgammonApp().run()
