"""Utility functions for backgammon AI."""

from config import Config


def move_to_indices(start, end):
    """
    Convert logical move (start, end) to action indices.
    
    Args:
        start: Board position (0-23), or 'bar'
        end: Board position (0-23), or 'off'
        
    Returns:
        (start_idx, end_idx) where bar=24, off=25
    """
    s = Config.BAR_IDX if start == 'bar' else start
    e = Config.OFF_IDX if end == 'off' else end
    return s, e


def indices_to_move(s, e):
    """
    Convert action indices back to logical move.
    
    Args:
        s: Start index (0-24, where 24=bar)
        e: End index (0-25, where 25=off)
        
    Returns:
        (start, end) as board positions or 'bar'/'off'
    """
    start = 'bar' if s == Config.BAR_IDX else s
    end = 'off' if e == Config.OFF_IDX else e
    return start, end


def flip_position(pos):
    """
    Flip a board position for canonical view (P-1 perspective).
    Position 0 becomes 23, position 23 becomes 0.
    
    Args:
        pos: Board position (0-23), 'bar', or 'off'
        
    Returns:
        Flipped position
    """
    if pos == 'bar' or pos == 'off':
        return pos
    return Config.NUM_POINTS - 1 - pos


def flip_action(action):
    """
    Flip an action for canonical view.
    
    Args:
        action: (start, end) tuple
        
    Returns:
        Flipped (start, end) tuple
    """
    start, end = action
    return (flip_position(start), flip_position(end))

