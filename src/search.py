"""Backwards-compatible re-export of the efficient MCTS search."""

from src.mcts import (  # noqa: F401
    ALL_ROLLS,
    Candidate,
    MCTS,
    SearchResult,
    Searcher,
    prune_losing_moves,
    select_candidate,
)
