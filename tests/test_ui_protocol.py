"""
Comprehensive checks for the web-UI protocol (scripts/play_web.py server).

Every action the HTML client can send is exercised here, in every order that
must be rejected as well as the order that must work:

    hello, new_game, new_match, roll, move, end_turn, double, take_double,
    refuse_double, set_mode, ai_play, load_model(bad)

Run from the repo root:  python3 -m tests.test_ui_protocol
(also pytest-compatible)
"""

import asyncio
import importlib.util
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("play_web", os.path.join(ROOT, "scripts", "play_web.py"))
play_web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(play_web)

from src.config import Config
from src.model import get_model
from src.search import Searcher
from src.utils.match_equity import MatchEquityTable

_ORIG_LOAD_MODEL = play_web.BackgammonServer._try_load_default_model
_ORIG_LOAD_EQ = play_web.BackgammonServer._try_load_equity_table


class FakeWS:
    """Collects every payload the server pushes."""
    def __init__(self):
        self.messages = []

    async def send(self, text):
        self.messages.append(json.loads(text))

    @property
    def last(self):
        return self.messages[-1] if self.messages else None

    def statuses(self):
        return [m["payload"]["status"] for m in self.messages if m.get("type") == "state"]


def make_server(mode="human_vs_ai", cube=True, target=7):
    play_web.BackgammonServer._try_load_default_model = lambda self: None
    play_web.BackgammonServer._try_load_equity_table = lambda self: None
    try:
        s = play_web.BackgammonServer()
    finally:
        play_web.BackgammonServer._try_load_default_model = _ORIG_LOAD_MODEL
        play_web.BackgammonServer._try_load_equity_table = _ORIG_LOAD_EQ
    s.game_mode = mode
    s.match_target = target
    s.game.match_target = target
    s.game.match_scores = {1: 0, -1: 0}
    return s


def to_human_turn(s, dice=None):
    """Put the server in a clean 'human (white) may act' state."""
    s.game_over = False
    s.game.turn = 1
    s.game.dice = list(dice) if dice else []
    s.has_rolled = bool(dice)
    s.opening_pending = False
    s.waiting_for_cube_decision = False
    s.game.cube_offered = False


# ---------------------------------------------------------------------
# Serialization / hello
# ---------------------------------------------------------------------
def test_hello_payload_fields():
    s = make_server()
    msg = s.serialize("hi")
    p = msg["payload"]
    for key in ("board", "bar", "off", "pips", "turn", "dice", "cube_value",
                "cube_owner", "can_double", "waiting_for_cube", "legal_moves",
                "status", "game_over", "winner", "mode", "model_loaded",
                "has_rolled", "match_target", "match_scores", "crawford"):
        assert key in p, key
    assert msg["type"] == "state"


def test_pips_match_engine():
    s = make_server()
    s.new_game()
    p = s.serialize()["payload"]
    assert p["pips"]["white"] == s.game.pip_count(1)
    assert p["pips"]["black"] == s.game.pip_count(-1)


def test_legal_moves_serialized_as_pairs():
    s = make_server()
    to_human_turn(s, dice=[6, 3])
    p = s.serialize()["payload"]
    assert p["legal_moves"] and all(len(m) == 2 for m in p["legal_moves"])
    # every serialized pair must be accepted by the engine
    pairs = {(src, dst) for (src, dst), _ in s.game.get_legal_moves()}
    assert {(m[0], m[1]) for m in p["legal_moves"]} == pairs


# ---------------------------------------------------------------------
# new game / new match
# ---------------------------------------------------------------------
def test_new_game_opening_state():
    s = make_server()
    p = s.new_game()["payload"]
    assert s.opening_pending and s.has_rolled and len(s.game.dice) == 2
    assert s.game.dice[0] != s.game.dice[1]          # opening dice differ
    assert "Opening roll" in p["status"]
    assert p["turn"] == s.game.turn


def test_new_game_and_match_reset_pending_cube():
    s = make_server()
    to_human_turn(s)
    s.waiting_for_cube_decision = True
    s.game.turn = -1
    s.game.cube_offered = True
    s.new_game()
    assert not s.waiting_for_cube_decision and not s.game.cube_offered
    s.waiting_for_cube_decision = True
    s.new_match(5)
    assert not s.waiting_for_cube_decision
    assert s.match_target == 5 and s.game.match_target == 5


def test_new_match_bad_target_is_clamped_or_kept():
    s = make_server(target=7)
    s.new_match("garbage")   # must not raise
    assert s.match_target == 7
    assert s.new_match(99)["payload"]["match_target"] == 21
    assert s.new_match(-3)["payload"]["match_target"] == 1


def test_match_scores_survive_new_game_reset_by_new_match():
    s = make_server(target=7)
    s.game.match_scores[1] = 3
    s.new_game()
    assert s.game.match_scores[1] == 3
    s.new_match(7)
    assert s.game.match_scores == {1: 0, -1: 0}


# ---------------------------------------------------------------------
# roll
# ---------------------------------------------------------------------
def test_roll_flow_and_guards():
    s = make_server(cube=False)
    to_human_turn(s)
    random.seed(42)
    p = s.roll()["payload"]
    assert s.has_rolled and s.game.dice
    assert "Rolled" in p["status"]
    # second roll blocked
    assert "Already rolled" in s.roll()["payload"]["status"]
    # end turn -> rolling again is allowed for the new turn
    to_human_turn(s)
    p = s.roll()["payload"]
    assert s.game.dice, "fresh turn must roll fresh dice"


def test_roll_blocked_on_ai_turn_and_during_cube():
    s = make_server()
    s.game_over = False
    s.game.turn = -1
    s.game.dice = []
    s.has_rolled = False
    assert "AI" in s.roll()["payload"]["status"]

    to_human_turn(s)
    s.waiting_for_cube_decision = True
    assert "double" in s.roll()["payload"]["status"].lower()


def test_roll_blocked_after_game_over():
    s = make_server()
    s.game_over = True
    assert "over" in s.roll()["payload"]["status"].lower()


# ---------------------------------------------------------------------
# move
# ---------------------------------------------------------------------
def test_move_requires_roll_and_legality():
    s = make_server()
    to_human_turn(s)
    assert "Roll dice first" in s.make_move(13, 8)["payload"]["status"]

    to_human_turn(s, dice=[6, 3])
    before = tuple(s.game.board)
    assert "Illegal move" in s.make_move(0, 12)["payload"]["status"]
    assert tuple(s.game.board) == before  # board untouched


def test_move_accepts_string_numbers():
    s = make_server()
    to_human_turn(s, dice=[6, 3])
    legal = s.game.get_legal_moves()
    (src, dst), _ = legal[0]
    before = sum(s.game.board)
    p = s.make_move(str(src), str(dst))["payload"]
    assert "Illegal" not in p["status"]
    assert sum(s.game.board) == before


def test_move_blocked_on_ai_turn_during_cube_and_after_over():
    s = make_server()
    to_human_turn(s, dice=[6, 3])
    s.game.turn = -1
    assert "AI" in s.make_move("bar", 3)["payload"]["status"]

    to_human_turn(s, dice=[6, 3])
    s.waiting_for_cube_decision = True
    assert "double" in s.make_move(13, 8)["payload"]["status"].lower()

    to_human_turn(s, dice=[6, 3])
    s.game_over = True
    assert "over" in s.make_move(13, 7)["payload"]["status"].lower()


def test_move_updates_bar_and_hit_state():
    s = make_server()
    to_human_turn(s, dice=[6, 3])
    # white 7 -> 1 hits a black blot at 1
    g = s.game
    g.board = [0] * 24
    g.board[7] = 1
    g.board[1] = -1
    g.turn = 1
    g.dice = [6]
    p = s.make_move(7, 1)["payload"]
    assert g.bar[1] == 1 and g.board[1] == 1 and g.board[7] == 0
    assert "end your turn" in p["status"] or "continue" in p["status"]
    assert p["dice"] == []


def test_bear_off_move_to_off_target():
    s = make_server()
    to_human_turn(s)
    g = s.game
    g.board = [0] * 24
    g.board[2] = 1
    g.off = [14, 0]
    g.turn = 1
    g.dice = [3]
    p = s.make_move(2, "off")["payload"]
    assert g.off[0] == 15 and s.game_over and p["game_over"]
    assert p["winner"] == 1
    assert "wins" in p["status"].lower()


# ---------------------------------------------------------------------
# end_turn
# ---------------------------------------------------------------------
def test_end_turn_guards():
    s = make_server()
    to_human_turn(s)
    assert "Roll dice first" in s.end_turn()["payload"]["status"]

    to_human_turn(s, dice=[6, 3])
    assert "legal move" in s.end_turn()["payload"]["status"]

    # fully blocked roll: end turn allowed (single white checker, its only
    # destination for the die is closed by two black checkers)
    to_human_turn(s, dice=[5])
    g = s.game
    g.board = [0] * 24
    g.board[13] = 1
    g.board[8] = -2                       # 13-5=8 blocked; no other white checker
    g.turn = 1
    g.dice = [5]
    p = s.end_turn()["payload"]
    assert "turn" in p["status"].lower()
    assert s.game.turn == -1 and not s.has_rolled


def test_end_turn_blocked_during_cube_and_ai_turn():
    s = make_server()
    to_human_turn(s, dice=[])
    s.waiting_for_cube_decision = True
    assert "double" in s.end_turn()["payload"]["status"].lower()
    s.waiting_for_cube_decision = False
    s.game.turn = -1
    assert "AI" in s.end_turn()["payload"]["status"]


# ---------------------------------------------------------------------
# doubling cube
# ---------------------------------------------------------------------
def test_double_disabled_when_cube_off():
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = False
    try:
        s = make_server()
        to_human_turn(s)
        assert not s.can_offer_double()
        msg = asyncio.run(s.offer_double(FakeWS()))
        assert msg is not None and "Cannot double" in msg["payload"]["status"]
    finally:
        Config.CUBE_ENABLED = old


def test_double_blocked_before_own_turn_or_after_roll_or_opening():
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    try:
        s = make_server()
        # opening roll pending: dice on table
        s.new_game()
        assert not s.can_offer_double()
        # after rolling
        to_human_turn(s, dice=[4, 2])
        assert not s.can_offer_double()
        # AI's turn: router rejects
        to_human_turn(s)
        s.game.turn = -1
        msg = asyncio.run(s.handle({"type": "double"}, FakeWS()))
        assert msg and "AI" in msg["payload"]["status"]
    finally:
        Config.CUBE_ENABLED = old


def _offer_state(s):
    """The pending state exactly as offer_double() leaves it (white offered,
    turn flipped to the black responder). Set directly so the human-vs-AI
    auto-answer in offer_double cannot race the manual take/refuse."""
    s.waiting_for_cube_decision = True
    s.game.turn = -1
    s.game.cube_offered = True


def test_double_take_flow():
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    try:
        s = make_server()
        to_human_turn(s)                          # white to roll
        ws = FakeWS()
        _offer_state(s)
        assert not s.can_offer_double()           # blocked while pending
        p = s.serialize()["payload"]
        assert p["waiting_for_cube"] and not p["can_double"]

        result = s.take_double()
        assert not s.waiting_for_cube_decision
        assert s.game.cube == 2 and s.game.cube_owner == -1
        assert s.game.turn == 1                   # doubler still rolls
        assert "accepted" in result["payload"]["status"].lower()
        # human still must roll afterwards
        random.seed(1)
        assert "Rolled" in s.roll()["payload"]["status"]
    finally:
        Config.CUBE_ENABLED = old


def test_double_offer_routes_to_ai_responder():
    """A human's real offer_double in human_vs_ai is answered by the AI inline."""
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    random.seed(3)
    try:
        s = make_server()                         # human white, AI black
        to_human_turn(s)
        ws = FakeWS()
        msg = asyncio.run(s.offer_double(ws))
        assert msg is None                        # resolved inline
        assert not s.waiting_for_cube_decision
        assert s.game.cube in (1, 2)              # took or dropped
        if s.game.cube == 1:
            assert s.game_over and s.winner == 1  # AI dropped: doubler wins
        else:
            assert s.game.cube_owner == -1 and s.game.turn == 1
    finally:
        Config.CUBE_ENABLED = old


def test_double_refuse_flow_scores_doubler():
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    try:
        s = make_server()
        to_human_turn(s)
        _offer_state(s)
        before = s.game.match_scores[1]
        cube_before = s.game.cube
        result = s.refuse_double()["payload"]
        assert s.game_over and s.winner == 1
        # doubler wins the pre-double cube value
        assert s.game.match_scores[1] == before + cube_before
        assert result["game_over"] and result["winner"] == 1
    finally:
        Config.CUBE_ENABLED = old


def test_take_refuse_without_pending():
    s = make_server()
    assert "No double offered" in s.take_double()["payload"]["status"]
    assert "No double offered" in s.refuse_double()["payload"]["status"]


def test_router_blocks_take_refuse_while_ai_is_responder():
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    try:
        s = make_server()
        to_human_turn(s)
        s.waiting_for_cube_decision = True
        s.game.turn = -1               # AI is the responder
        s.game.cube_offered = True
        msg = asyncio.run(s.handle({"type": "take_double"}, FakeWS()))
        assert msg and "AI is responding" in msg["payload"]["status"]
        assert s.waiting_for_cube_decision      # unchanged
        msg = asyncio.run(s.handle({"type": "refuse_double"}, FakeWS()))
        assert msg and "AI is responding" in msg["payload"]["status"]
        assert s.waiting_for_cube_decision
    finally:
        Config.CUBE_ENABLED = old


def test_set_mode_blocked_during_pending_cube():
    s = make_server()
    s.waiting_for_cube_decision = True
    msg = asyncio.run(s.handle({"type": "set_mode", "mode": "ai_vs_ai"}, FakeWS()))
    assert msg and "double" in msg["payload"]["status"].lower()
    assert s.game_mode != "ai_vs_ai"


def test_double_owner_blocks_redouble():
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    try:
        s = make_server()
        to_human_turn(s)
        s.game.cube_owner = -1          # black owns the cube; white may not double
        assert not s.can_offer_double()
    finally:
        Config.CUBE_ENABLED = old


# ---------------------------------------------------------------------
# game-over lock
# ---------------------------------------------------------------------
def test_all_actions_locked_after_game_over():
    s = make_server()
    s.game_over = True
    s.winner = 1
    assert "over" in s.roll()["payload"]["status"].lower()
    assert "over" in s.end_turn()["payload"]["status"].lower()
    assert "over" in s.make_move(13, 8)["payload"]["status"].lower()
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    try:
        assert not s.can_offer_double()
    finally:
        Config.CUBE_ENABLED = old


# ---------------------------------------------------------------------
# router unknown / model error paths
# ---------------------------------------------------------------------
def test_router_unknown_command():
    s = make_server()
    msg = asyncio.run(s.handle({"type": "nope"}, FakeWS()))
    assert "Unknown command" in msg["payload"]["status"]


def test_load_model_garbage_reports_error():
    s = make_server()
    msg = s.load_model_from_data("bogus.pt", "aGVsbG8=")  # base64 "hello"
    assert msg["type"] == "model_error"


# ---------------------------------------------------------------------
# End-to-end: the actual bot plays legal turns via the router
# ---------------------------------------------------------------------
def test_ai_never_makes_illegal_moves(turns=12):
    old_device = Config.DEVICE
    Config.DEVICE = "cpu"
    old = Config.CUBE_ENABLED
    Config.CUBE_ENABLED = True
    random.seed(11)
    try:
        s = make_server(mode="ai_vs_ai")
        model = get_model().to("cpu").eval()
        s.model = model
        s.mcts = Searcher(model, device="cpu", equity_table=MatchEquityTable())
        ws = FakeWS()
        s.new_match(21)

        async def drive():
            for _ in range(turns):
                if s.game_over:
                    await s.handle({"type": "new_game"}, ws)
                await s.handle({"type": "ai_play"}, ws)
        asyncio.run(drive())

        for m in ws.messages:
            assert m["type"] in ("state", "error"), m
            if m["type"] == "error":
                raise AssertionError(f"server error during AI play: {m}")
            st = m["payload"]["status"]
            assert "AI Error" not in st, st
            assert "Illegal" not in st, st
            assert "Engine Error" not in st, st
            assert "could not be applied" not in st, st

        # checker conservation survives whatever sequence the AI played
        g = s.game
        w = sum(x for x in g.board if x > 0) + g.bar[0] + g.off[0]
        b = sum(-x for x in g.board if x < 0) + g.bar[1] + g.off[1]
        assert w == 15 and b == 15, (w, b)
        assert any("AI rolled" in m["payload"]["status"] for m in ws.messages)
    finally:
        Config.CUBE_ENABLED = old
        Config.DEVICE = old_device


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} UI-protocol tests passed")
