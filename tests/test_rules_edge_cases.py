"""
Comprehensive engine legality checks (rules edge cases + AI execution route).

Run from the repo root:  python3 -m tests.test_rules_edge_cases
(also pytest-compatible)

The critical regression guarded here: the live bot plans a turn with
`get_legal_turns()` (via MCTS) but executes each atomic leg with
`step_atomic()`, which validates against `get_legal_moves()`. If the two
maximality filters disagree, the bot plays (or is blocked from playing)
"illegal" moves mid-turn.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import BackgammonGame
from src.config import Config


def new_game():
    g = BackgammonGame(train_mode=False)
    g.reset()
    return g


def set_pos(g, board=None, bar=(0, 0), off=(0, 0), turn=1, dice=(), blacks=None):
    """Place a legal position: white checkers via `board` dict, black via `blacks`."""
    g.reset()
    g.board = [0] * Config.NUM_POINTS
    for k, v in (board or {}).items():
        g.board[k] = v
    for k, v in (blacks or {}).items():
        g.board[k] = v
    g.bar = list(bar)
    g.off = list(off)
    g.turn = turn
    g.dice = list(dice)
    return g


def first_legs(g):
    return {path[0] for path, _ in g.get_legal_turns()}


def atomic_actions(g):
    return {tuple(a) for a in g.get_legal_moves()}


def replay_every_turn(g):
    """Every turn get_legal_turns offers must replay leg-by-leg via step_atomic."""
    for path, final_pos in g.get_legal_turns():
        h = g.copy()
        for leg in path:
            h.step_atomic(leg)
        assert (tuple(h.board), tuple(h.bar), tuple(h.off)) == final_pos, (
            f"turn {path} replays to different position than get_legal_turns promised"
        )


# ---------------------------------------------------------------------
# Bar entry
# ---------------------------------------------------------------------
def test_bar_entry_direction():
    g = new_game()
    set_pos(g, bar=(1, 0), turn=1, dice=[3])
    assert g.get_legal_moves() == [(("bar", 21), 3)], g.get_legal_moves()

    set_pos(g, bar=(0, 1), turn=-1, dice=[3])
    assert g.get_legal_moves() == [(("bar", 2), 3)], g.get_legal_moves()


def test_bar_entry_blocked():
    g = new_game()
    # white must enter on 24-die; two black checkers close it
    set_pos(g, blacks={21: -2}, bar=(1, 0), turn=1, dice=[3])
    assert g.get_legal_moves() == []

    # a blot does NOT block entry: entering hits it
    set_pos(g, blacks={21: -1}, bar=(1, 0), turn=1, dice=[3])
    moves = g.get_legal_moves()
    assert (("bar", 21), 3) in moves, moves
    h = g.copy()
    h.step_atomic((("bar", 21), 3))
    assert h.bar == [0, 1] and h.board[21] == 1, (h.board[21], h.bar)


def test_bar_forces_entry_only():
    g = new_game()
    set_pos(g, board={7: 1}, bar=(1, 0), turn=1, dice=[2, 5])
    firsts = atomic_actions(g)
    assert all(src == "bar" for (src, _), _ in firsts), firsts


def test_bar_partial_entry_with_two_dice():
    g = new_game()
    # only the 5 die can enter (24-2=22 closed, 24-5=19 open)
    set_pos(g, blacks={22: -2}, bar=(1, 0), turn=1, dice=[2, 5])
    assert atomic_actions(g) == {(("bar", 19), 5)}, atomic_actions(g)


# ---------------------------------------------------------------------
# Movement, hits, blocking
# ---------------------------------------------------------------------
def test_hit_and_block():
    g = new_game()
    set_pos(g, board={10: 1}, blacks={7: -1}, turn=1, dice=[3])
    assert ((10, 7), 3) in atomic_actions(g)

    set_pos(g, board={10: 1}, blacks={7: -2}, turn=1, dice=[3])
    assert atomic_actions(g) == set(), atomic_actions(g)


def test_own_stacking_allowed():
    g = new_game()
    set_pos(g, board={10: 2}, turn=1, dice=[3])
    assert ((10, 7), 3) in atomic_actions(g)


def test_doubles_consume_four_dice():
    g = new_game()
    set_pos(g, board={23: 4}, turn=1, dice=[6, 6, 6, 6])
    h = g.copy()
    for _ in range(4):
        h.step_atomic(((23, 17), 6))
    assert h.dice == [] and h.board[17] == 4


# ---------------------------------------------------------------------
# Maximality
# ---------------------------------------------------------------------
def test_must_use_both_dice_when_possible():
    g = new_game()
    # single checker on 8, open board: both 5+3 orderings play all dice;
    # a one-dice turn must never be offered.
    set_pos(g, board={8: 1}, turn=1, dice=[3, 5])
    turns = g.get_legal_turns()
    assert turns, "expected legal turns"
    assert all(len(p) == 2 for p, _ in turns), turns
    replay_every_turn(g)


def test_partial_play_prefers_larger_die():
    g = new_game()
    # One checker on the bar: entry with the 3 lands on 21 and the 5 cannot
    # continue (16 closed); entry with the 5 lands on 19 and the 3 cannot
    # continue (16 closed). Both alternatives play exactly one die, so the
    # larger die (5) must be preferred by the maximality pip rule.
    set_pos(g, blacks={16: -2}, bar=(1, 0), turn=1, dice=[3, 5])
    assert atomic_actions(g) == {(("bar", 19), 5)}, atomic_actions(g)
    assert first_legs(g) == atomic_actions(g), first_legs(g)


# ---------------------------------------------------------------------
# Bear off
# ---------------------------------------------------------------------
def test_bear_off_requires_home_board():
    g = new_game()
    # white checker on index 6 (outside home 0-5): no overshoot bear-off anywhere
    set_pos(g, board={0: 1, 6: 1}, turn=1, dice=[6])
    assert all(dst != "off" or src == 0 for (src, dst) in
               [m for m, _ in g.get_legal_moves()]), g.get_legal_moves()
    # the checker at 0 (dist 1) with die 6: overshoot only if it's furthest —
    # it isn't (index 6 holds a checker), so only 6->0 ... impossible; must move 6->0
    assert atomic_actions(g) == {((6, 0), 6)}, atomic_actions(g)


def test_bear_off_exact_and_overshoot():
    g = new_game()
    set_pos(g, board={2: 1}, off=(14, 15), turn=1, dice=[3])
    assert ((2, "off"), 3) in atomic_actions(g)

    set_pos(g, board={2: 1}, off=(14, 15), turn=1, dice=[5])
    assert ((2, "off"), 5) in atomic_actions(g)  # overshoot, furthest checker

    set_pos(g, board={1: 1, 2: 1}, off=(13, 15), turn=1, dice=[5])
    # checker on 2 may NOT overshoot while checker on 1... wait: further from
    # edge = higher index for white. Index 2 is further, so index 1 blocks it?
    # No: from edge means larger index for white. The checker at 2 is the
    # furthest; 1 is nearer the edge, so bearing 2 off with 5 is allowed and
    # bearing 1 off with 5 is not (a checker is further back at 2).
    firsts = atomic_actions(g)
    assert ((2, "off"), 5) in firsts
    assert ((1, "off"), 5) not in firsts


def test_bar_blocks_bear_off():
    g = new_game()
    set_pos(g, board={2: 1}, bar=(1, 0), off=(14, 15), turn=1, dice=[3, 5])
    assert all(dst != "off" for (src, dst) in [m for m, _ in g.get_legal_moves()])
    assert all(src == "bar" for (src, _), _ in g.get_legal_moves())


# ---------------------------------------------------------------------
# THE regression: bear-off dice ambiguity must agree between the two
# generators (single checker at index 0, dice [4,3] -> die 4 must be used)
# ---------------------------------------------------------------------
def test_bearoff_die_ambiguity_consistency():
    g = new_game()
    for dice in ([4, 3], [3, 4], [6, 5], [2, 1]):
        set_pos(g, board={0: 1}, off=(14, 15), turn=1, dice=list(dice))
        big = max(dice)
        assert ((0, "off"), big) in atomic_actions(g)
        assert first_legs(g) <= atomic_actions(g), (
            f"get_legal_turns offers a leg step_atomic will reject (dice {dice})"
        )
        replay_every_turn(g)


def test_turn_legs_replay_on_dense_positions():
    g = new_game()
    cases = [
        dict(board={23: 2, 12: 5, 7: 3, 5: 5}, turn=1, dice=[3, 3, 3, 3]),
        dict(board={18: 1, 19: 1}, blacks={23: -1, 0: -14}, turn=1, dice=[1, 2]),
        dict(board={5: 15}, turn=1, dice=[1, 1, 1, 1]),
        dict(blacks={0: 2, 11: 5, 16: 3, 18: 5}, turn=-1, dice=[6, 6, 6, 6]),
        dict(board={0: 1, 1: 1}, off=(13, 15), turn=1, dice=[2, 1]),
        dict(board={1: 1}, off=(14, 15), turn=1, dice=[2, 2, 2, 2]),
    ]
    for c in cases:
        set_pos(g, **c)
        replay_every_turn(g)
        assert first_legs(g) <= atomic_actions(g), c


# ---------------------------------------------------------------------
# step_atomic rejects invalid actions
# ---------------------------------------------------------------------
def test_step_atomic_rejections():
    g = new_game()
    set_pos(g, board={10: 1}, blacks={7: -2}, turn=1, dice=[3])
    for bad in (
        ((10, 7), 3),      # blocked point
        ((10, 8), 2),      # no checker can move that die here (die/order mismatch too)
        ((11, 8), 3),      # no white checker on 11
        (("bar", 21), 3),  # nothing on the bar
    ):
        try:
            g.step_atomic(bad)
            raise AssertionError(f"step_atomic accepted {bad}")
        except ValueError:
            pass
    # die-mismatch is rejected even for an otherwise legal (src, dst)
    set_pos(g, board={10: 1}, turn=1, dice=[3, 5])
    try:
        g.step_atomic(((10, 5), 3))  # 5 pips away but claims the 3-die
        raise AssertionError("die mismatch accepted")
    except ValueError:
        pass


# ---------------------------------------------------------------------
# Win detection / scoring (train_mode=False: 1/2/3)
# ---------------------------------------------------------------------
def test_win_type_detection():
    g = new_game()
    set_pos(g, board={0: 1}, off=(14, 1), blacks={20: -14}, turn=1, dice=[1])
    assert g.win_type(1) == 1  # loser has borne off -> single

    set_pos(g, board={0: 1}, off=(14, 0), blacks={20: -15}, turn=1, dice=[1])
    assert g.win_type(1) == 2  # loser off=0, no bar, none in white home -> gammon

    set_pos(g, board={0: 1}, off=(14, 0), blacks={3: -1, 20: -14}, turn=1, dice=[1])
    assert g.win_type(1) == 3  # loser checker in winner's home (0-5) -> backgammon

    set_pos(g, board={0: 1}, bar=(0, 1), off=(14, 0), blacks={20: -14}, turn=1, dice=[1])
    assert g.win_type(1) == 3  # loser on the bar -> backgammon

    # mirror for black winning: black home is 18-23
    set_pos(g, board={8: 14, 23: -1}, off=(0, 14), turn=-1, dice=[1])
    assert g.win_type(-1) == 2  # white never bore off, none in black home -> gammon
    g.board[19] = 1
    g.board[8] = 13
    assert g.win_type(-1) == 3  # white checker in black's home -> backgammon


def test_final_scoring_adds_cube_times_type():
    g = new_game()
    set_pos(g, board={0: 1}, off=(14, 0), turn=1, dice=[1])
    g.cube = 4  # set after set_pos (reset() restores cube=1)
    before = g.match_scores[1]
    winner, points = g.step_atomic(((0, "off"), 1))
    assert winner == 1 and points == 8 and g.match_scores[1] == before + 8


# ---------------------------------------------------------------------
# Opening roll
# ---------------------------------------------------------------------
def test_opening_roll():
    random.seed(1234)
    g = new_game()
    dice = g.roll_opening()
    assert dice[0] != dice[1]
    assert g.turn == (1 if dice[0] > dice[1] else -1)
    assert g.dice == dice


# ---------------------------------------------------------------------
# Random games: AI execution route + invariants
# ---------------------------------------------------------------------
def _count(board, pl):
    return sum(abs(x) for x in board if (x > 0 if pl == 1 else x < 0))


def test_random_games_ai_route(games=12, seed=7, max_seconds=45):
    import time
    random.seed(seed)
    g = BackgammonGame(train_mode=False)
    t0 = time.time()
    done = 0
    while done < games and time.time() - t0 < max_seconds:
        g.reset()
        g.match_scores = {1: 0, -1: 0}
        g.roll_opening()
        for _ in range(500):
            if g.check_win()[0]:
                break
            if not g.dice:
                g.roll_dice()
            turns = g.get_legal_turns()
            legal_firsts = atomic_actions(g)
            # every offered full turn must start with an atomic-legal action
            assert first_legs(g) <= legal_firsts, (g.board, g.bar, g.off, g.dice)
            if not turns:
                assert not legal_firsts
                g.switch_turn()
                continue
            path = random.choice(turns)[0]
            for leg in path:
                w = _count(g.board, 1) + g.bar[0] + g.off[0]
                b = _count(g.board, -1) + g.bar[1] + g.off[1]
                g.step_atomic(leg)
                assert _count(g.board, 1) + g.bar[0] + g.off[0] == w == 15
                assert _count(g.board, -1) + g.bar[1] + g.off[1] == b == 15
            if g.check_win()[0]:
                break
            g.switch_turn()
        done += 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} engine rule tests passed")
