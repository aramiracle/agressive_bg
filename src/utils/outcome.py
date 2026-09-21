"""
Outcome-distribution helpers.

The value head predicts a distribution over the six ways a game can end, from
the perspective of the player the position is encoded for:

    index 0: win  single      3: lose single
    index 1: win  gammon      4: lose gammon
    index 2: win  backgammon  5: lose backgammon

Everything downstream (search equity, cube decisions, TD targets) is derived
from this distribution, so the network never has to learn a scalar that mixes
game outcome with match-score context.
"""

import math
import torch
from src.config import Config

WIN_SINGLE, WIN_GAMMON, WIN_BACKGAMMON = 0, 1, 2
LOSE_SINGLE, LOSE_GAMMON, LOSE_BACKGAMMON = 3, 4, 5

# win_type: 1 = single, 2 = gammon, 3 = backgammon
_WIN_TYPE_TO_IDX = {1: WIN_SINGLE, 2: WIN_GAMMON, 3: WIN_BACKGAMMON}


def point_multipliers():
    """Points a single/gammon/backgammon is worth (before the cube)."""
    if Config.TRAIN_MODE:
        return (Config.R_WIN, Config.R_GAMMON, Config.R_BACKGAMMON)
    return (1.0, 2.0, 3.0)


def money_weights():
    """Signed per-outcome weights, normalised so equity lies in [-1, 1]."""
    m = point_multipliers()
    scale = max(m)
    return torch.tensor(
        [m[0], m[1], m[2], -m[0], -m[1], -m[2]], dtype=torch.float32
    ) / scale


def money_equity(probs):
    """Normalised cubeless equity in [-1, 1]. probs: [..., 6]."""
    return (probs * money_weights().to(probs.device)).sum(-1)


def flip(probs):
    """Swap perspective: my win outcomes become the opponent's losses."""
    return torch.cat([probs[..., 3:6], probs[..., 0:3]], dim=-1)


def terminal_distribution(won, win_type):
    """One-hot outcome for a finished game from the perspective of one player."""
    t = torch.zeros(Config.NUM_OUTCOMES, dtype=torch.float32)
    idx = _WIN_TYPE_TO_IDX[win_type]
    t[idx if won else idx + 3] = 1.0
    return t


def race_distribution(p_win):
    """Pure race with no gammon possible: only single win / single loss."""
    t = torch.zeros(Config.NUM_OUTCOMES, dtype=torch.float32)
    t[WIN_SINGLE] = p_win
    t[LOSE_SINGLE] = 1.0 - p_win
    return t


def race_win_probability(pips_on_roll, pips_opponent):
    """
    Win probability of the player on roll in a no-contact race.

    Normal approximation: each roll moves ~8.17 pips with std ~4.3, and being
    on roll is worth about half a roll (4 pips). Bear-off wastage is ignored.
    """
    lead = pips_opponent - pips_on_roll + 4.0
    total = max(pips_on_roll + pips_opponent, 1.0)
    z = 0.665 * lead / math.sqrt(total)
    p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return min(0.999, max(0.001, p))


def match_equity_from_outcomes(probs, my_score, opp_score, cube, equity_table):
    """
    Expected match-winning probability in [0, 1] given an outcome distribution,
    the current score, and the current cube value.
    """
    target = Config.MATCH_TARGET
    mults = point_multipliers()
    me = 0.0
    p = probs.tolist()
    for k, m in enumerate(mults):
        pts = int(round(cube * m))
        me += p[k] * equity_table.get_equity(min(my_score + pts, target), opp_score)
        me += p[k + 3] * equity_table.get_equity(my_score, min(opp_score + pts, target))
    return me


def match_equity_batch(probs, my_score, opp_score, cube, equity_table):
    """
    Vectorised version of match_equity_from_outcomes for a [N, 6] tensor.
    Returns match equity mapped to [-1, 1] so it is comparable to money_equity.
    """
    target = Config.MATCH_TARGET
    mults = point_multipliers()
    w = []
    for m in mults:
        pts = int(round(cube * m))
        w.append(equity_table.get_equity(min(my_score + pts, target), opp_score))
    for m in mults:
        pts = int(round(cube * m))
        w.append(equity_table.get_equity(my_score, min(opp_score + pts, target)))
    weights = torch.tensor(w, dtype=torch.float32, device=probs.device)
    return 2.0 * (probs * weights).sum(-1) - 1.0
