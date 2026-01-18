"""Utility functions for backgammon AI."""

from .config import Config


def move_to_indices(start, end):
    """
    Convert a move (start, end) to policy head indices.
    
    Args:
        start: Starting position ('bar' or 0-23)
        end: Ending position ('off' or 0-23)
    
    Returns:
        Tuple of (start_idx, end_idx) for policy heads
    """
    if start == 'bar':
        start_idx = Config.BAR_IDX
    else:
        start_idx = start
    
    if end == 'off':
        end_idx = Config.OFF_IDX
    else:
        end_idx = end
    
    return start_idx, end_idx


def indices_to_move(start_idx, end_idx):
    """
    Convert policy head indices back to a move tuple.
    
    Args:
        start_idx: Start index from policy head
        end_idx: End index from policy head
    
    Returns:
        Tuple of (start, end) where start/end can be 'bar'/'off' or int
    """
    if start_idx == Config.BAR_IDX:
        start = 'bar'
    else:
        start = start_idx
    
    if end_idx == Config.OFF_IDX:
        end = 'off'
    else:
        end = end_idx
    
    return start, end


def format_move(move):
    """Format a move for display."""
    start, end = move
    start_str = "BAR" if start == 'bar' else str(start + 1)
    end_str = "OFF" if end == 'off' else str(end + 1)
    return f"{start_str}->{end_str}"


def format_board(board, bar, off):
    """Format board state for display."""
    lines = []
    
    # Top half (points 13-24)
    top = " ".join(f"{board[i]:+3d}" for i in range(12, 24))
    lines.append(f"13-24: {top}")
    
    # Bottom half (points 1-12, reversed for display)
    bot = " ".join(f"{board[i]:+3d}" for i in range(11, -1, -1))
    lines.append(f" 1-12: {bot}")
    
    # Bar and off
    lines.append(f"Bar: W={bar[0]} B={bar[1]} | Off: W={off[0]} B={off[1]}")
    
    return "\n".join(lines)

