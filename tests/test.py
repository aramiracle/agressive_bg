"""
Regression suite for the value-search training pipeline.

Run from the repo root:  python -m tests.test
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.engine import BackgammonGame
from src.search import Searcher, select_candidate
from src.mcts import Candidate, prune_losing_moves
from src.model import get_model, LegacyValueTransformer
from src.utils.checkpoint import load_model_with_config, baseline_artifact_paths, save_checkpoint
from src.config import Config
from src.utils.train import train_batch
from src.utils.outcome import flip, money_equity, terminal_distribution, race_win_probability
from src.utils.game import _assign_targets, play_self_play_match
from src.utils.match_equity import MatchEquityTable
from src.utils.elo import (
    mixed_opponent_elo, passes_gate, play_single_game as eval_play_game,
    plays_against_baseline, promoted_elo, split_eval_games, update_elo,
)
from src.utils.cube import cube_decision_gain
from src.utils.outcome import match_equity_from_outcomes
from src.trainer_vs_baseline import split_matches
from src.replay_buffer import SimpleReplayBuffer


def check(cond, ok, bad):
    print(("✅ " + ok) if cond else ("❌ " + bad))
    return cond


def run_regression_suite():
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = get_model().to(device).eval()
    game = BackgammonGame()
    all_ok = True

    print("🚀 REGRESSION SUITE\n" + "=" * 40)

    print("\n[1] Model output shapes")
    board_t, ctx_t = game.get_vector(0, 0, device=device, canonical=True)
    out, cube = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
    all_ok &= check(out.shape == (1, Config.NUM_OUTCOMES) and cube.shape == (1, 2),
                    f"outcome {tuple(out.shape)}, cube {tuple(cube.shape)}",
                    f"unexpected shapes {tuple(out.shape)} / {tuple(cube.shape)}")
    all_ok &= check(ctx_t.shape[0] == Config.CONTEXT_SIZE,
                    f"context has {Config.CONTEXT_SIZE} features (cube owner + crawford included)",
                    "context size mismatch")

    print("\n[2] Canonical perspective symmetry")
    game.reset(); game.board[23] = 2; game.turn = 1
    v1, _ = game.get_vector(0, 0, canonical=True)
    game.reset(); game.board[0] = -2; game.turn = -1
    v2, _ = game.get_vector(0, 0, canonical=True)
    all_ok &= check(torch.equal(v1, v2), "board is identical from both perspectives",
                    "symmetry broken")

    print("\n[3] Outcome helpers")
    t = terminal_distribution(True, 2)
    all_ok &= check(float(money_equity(t)) > 0 and float(money_equity(flip(t))) < 0,
                    "flip() negates equity", "flip()/equity inconsistent")
    p = race_win_probability(100, 100)
    all_ok &= check(0.5 < p < 0.65, f"race formula: on roll in even 100-pip race = {p:.3f}",
                    f"race formula off: {p:.3f}")

    print("\n[4] Full-turn enumeration")
    game.reset(); game.turn = 1; game.dice = [3, 1]
    turns = game.get_legal_turns()
    all_ok &= check(len(turns) > 5 and all(len(path) == 2 for path, _ in turns),
                    f"{len(turns)} distinct complete plays for 3-1, all use both dice",
                    "enumeration failed")
    snap = game.fast_save()
    game.apply_turn(turns[0][0])
    all_ok &= check(game.dice == [] and (tuple(game.board), tuple(game.bar), tuple(game.off)) == turns[0][1],
                    "apply_turn reproduces the enumerated position", "apply_turn mismatch")
    game.fast_restore(snap)

    print("\n[5] MCTS (1-ply and 2-ply) is consistent and side-effect free")
    s1 = Searcher(model, ply=1)
    s2 = Searcher(model, ply=2)
    before = game.fast_save()
    r1 = s1.search(game, 0, 0)
    r2 = s2.search(game, 0, 0)
    all_ok &= check(game.fast_save() == before, "game state restored after search",
                    "search mutated the game")
    all_ok &= check(len(r1) == len(turns) and len(r2) == len(turns),
                    "every legal play has a candidate", "candidate count mismatch")
    probs_ok = all(abs(float(c.probs.sum()) - 1.0) < 1e-4 for c in r2.candidates)
    all_ok &= check(probs_ok, "candidate outcome distributions sum to 1", "bad distributions")
    visited = [c for c in r1.candidates if c.visits > 0]
    all_ok &= check(0 < len(visited) <= Config.SEARCH_PRUNE_TOP_K,
                    f"B1 prune: {len(visited)} survivors (≤ top_k={Config.SEARCH_PRUNE_TOP_K})",
                    f"prune kept {len(visited)} plays")
    all_ok &= check(
        all(abs(c.value() - c.equity) < 1e-5 for c in visited),
        "A1: same-player backup, Q equals leaf equity (never negated)",
        "A1 broken: child Q != leaf equity",
    )
    # terminal afterstate must be evaluated exactly (not by the network)
    game.reset(); game.board = [0] * 24; game.board[0] = 1; game.board[23] = -1
    game.off = [14, 14]; game.turn = 1; game.dice = [1, 2]
    rt = s2.search(game, 0, 0).best()
    all_ok &= check(float(rt.probs[0]) == 1.0, "bearing off the last checker = certain win",
                    f"terminal not exact: {rt.probs}")
    game.fast_restore(snap)

    print("\n[6] Exploration")
    picks = {id(select_candidate(r1, explore=True, temperature=1.0)) for _ in range(30)}
    all_ok &= check(len(picks) > 1, "high temperature samples different plays",
                    "exploration never deviates")
    all_ok &= check(select_candidate(r1, explore=False) is r1.best(), "greedy picks best",
                    "greedy did not pick best")

    print("\n[7] TD(lambda) targets")
    est = torch.tensor([0.5, 0.1, 0.0, 0.3, 0.1, 0.0])
    hist = [
        {'board': board_t, 'ctx': ctx_t, 'turn': 1, 'is_p1': True, 'is_cube': True,
         'cube_probs': torch.tensor([0.4, 0.6]), 'search_probs': None},
        {'board': board_t, 'ctx': ctx_t, 'turn': 1, 'is_p1': True, 'is_cube': False,
         'cube_probs': None, 'search_probs': est},
        {'board': board_t, 'ctx': ctx_t, 'turn': -1, 'is_p1': False, 'is_cube': False,
         'cube_probs': None, 'search_probs': est},
        {'board': board_t, 'ctx': ctx_t, 'turn': 1, 'is_p1': True, 'is_cube': False,
         'cube_probs': None, 'search_probs': est},
    ]
    data = _assign_targets(hist, winner=1, win_type=1)
    lam = Config.TD_LAMBDA
    term = terminal_distribution(True, 1)
    expected_first_move = (1 - lam) * est + lam * term  # bootstraps from next own decision
    all_ok &= check(torch.equal(data[3][2], term), "last decision of the winner -> exact win",
                    "terminal target wrong")
    all_ok &= check(torch.allclose(data[1][2], expected_first_move),
                    "earlier decision mixes next search estimate and later target",
                    "TD mixing wrong")
    all_ok &= check(torch.equal(data[0][2], data[1][2]) and data[0][3] is True,
                    "cube decision shares the following move's target",
                    "cube target wrong")
    all_ok &= check(torch.equal(data[2][2], terminal_distribution(False, 1)),
                    "loser's last decision -> exact loss", "loser terminal wrong")

    print("\n[8] Full self-play game + vectorised training step")
    Config.SEARCH_PLY = 1
    searcher = Searcher(model, ply=1)
    table = MatchEquityTable(match_target=Config.MATCH_TARGET)
    old_max_moves = Config.MAX_GAME_MOVES
    try:
        Config.MAX_GAME_MOVES = 0
        _, cap_winner, _ = play_self_play_match(game, searcher, model, device, table)
        all_ok &= check(cap_winner in (1, -1),
                        "move-cap abort still names a winner (no KeyError on scores[0])",
                        f"move-cap winner was {cap_winner}")
        Config.MAX_GAME_MOVES = 80
        data, winner, stats = play_self_play_match(game, searcher, model, device, table)
    finally:
        Config.MAX_GAME_MOVES = old_max_moves
    all_ok &= check(len(data) > 10 and winner in (1, -1),
                    f"self-play game finished: {len(data)} samples, {stats['games']} game(s)",
                    "self-play failed")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = SimpleReplayBuffer(10000)
    buffer.extend(data)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    losses = [train_batch(model, optimizer, buffer, min(32, len(data)), device, scaler)[0]
              for _ in range(8)]
    all_ok &= check(losses[-1] < losses[0], f"loss decreased {losses[0]:.4f} -> {losses[-1]:.4f}",
                    f"loss did not decrease {losses}")

    print("\n[9] Prune losing moves, cube stage, E3 gate")
    fake = [Candidate((), None) for _ in range(6)]
    for c, e in zip(fake, [0.90, 0.85, 0.40, 0.30, 0.20, -0.50]):
        c.equity = e
    kept = prune_losing_moves(fake, top_k=4, margin=0.10)
    all_ok &= check([round(c.equity, 2) for c in kept] == [0.90, 0.85],
                    "B1: only plays within 0.10 of the best survive prune",
                    f"prune kept {[c.equity for c in kept]}")

    old_cube, old_target = Config.CUBE_ENABLED, Config.MATCH_TARGET
    Config.CUBE_ENABLED = False
    g1 = BackgammonGame()
    g1.match_target = 1
    all_ok &= check(not g1.can_double(), "A4: stage-1 cube disabled", "cube still offered at target 1")
    Config.CUBE_ENABLED = True
    Config.MATCH_TARGET = 7
    g7 = BackgammonGame()
    g7.match_target = 7
    g7.set_match_scores(0, 0)
    all_ok &= check(g7.can_double(), "A4: 7-point opening can double", "cube blocked at 7-pt 0-0")
    g7.roll_opening()
    all_ok &= check(not g7.can_double() and g7.must_play_dice(),
                    "opening roll is played; the cube waits until the next turn",
                    "cube offered with the opening dice already rolled")
    g7.switch_turn()
    all_ok &= check(g7.can_double(),
                    "next turn can double before rolling",
                    "cube stayed closed after the opening roll was played")
    Config.CUBE_ENABLED = old_cube
    Config.MATCH_TARGET = old_target

    old_rate = Config.GATE_WIN_RATE
    Config.GATE_WIN_RATE = 0.525
    keep, _ = passes_gate(21, 40, 21, 40)
    promote, _ = passes_gate(22, 40, 22, 40)
    all_ok &= check(keep is False, "E3: 21/40 ≤ 52.5% keeps best_model", "promoted at 52.5%")
    all_ok &= check(promote is True, "E3: 22/40 > 52.5% promotes", "did not promote above 52.5%")
    Config.GATE_WIN_RATE = old_rate

    lagged = update_elo(511.0, 520.7, 22, 40)
    all_ok &= check(round(lagged) == 514 and max(520.7, lagged) == 520.7,
                    "22/40 from a lagged 511 stays under the 521 champion",
                    f"lagged rating {lagged} would have moved the champion")
    raised = promoted_elo(520.7, 22, 40)
    all_ok &= check(abs(raised - 522.7) < 1e-9,
                    "promotion rates the new best from the champion it beat (520.7 -> 522.7)",
                    f"promoted elo {raised}")

    all_ok &= check(not plays_against_baseline(True, 400.0, 600.0, 500.0),
                    "a lagged running elo keeps training on self-play while best is ahead",
                    "lagged running elo switched training onto the baseline")
    all_ok &= check(plays_against_baseline(True, 600.0, 400.0, 500.0),
                    "best elo under baseline enables training vs baseline",
                    "best elo under baseline stayed on self-play")
    all_ok &= check(not plays_against_baseline(True, 500.0, 500.0, 500.0),
                    "matching baseline elo collects against best",
                    "equal elo still played the baseline")
    all_ok &= check(not plays_against_baseline(False, 100.0, 100.0, 500.0),
                    "no baseline loaded stays off baseline games",
                    "baseline games enabled without a baseline")

    n_best, n_base = split_eval_games(40, best_elo=100.0, baseline_elo=500.0, has_baseline=True)
    all_ok &= check((n_best, n_base) == (20, 20),
                    "eval splits 20/20 vs best/baseline while best is weaker",
                    f"split was {n_best}/{n_base}")
    n_best, n_base = split_eval_games(40, best_elo=500.0, baseline_elo=500.0, has_baseline=True)
    all_ok &= check((n_best, n_base) == (40, 0),
                    "eval is all vs best once best matches baseline",
                    f"split was {n_best}/{n_base}")
    n_best, n_base = split_eval_games(40, best_elo=100.0, baseline_elo=500.0, has_baseline=False)
    all_ok &= check((n_best, n_base) == (40, 0),
                    "eval is all vs best when no baseline is loaded",
                    f"split was {n_best}/{n_base}")
    avg_elo = mixed_opponent_elo(100.0, 500.0, 20, 20)
    all_ok &= check(avg_elo == 300.0,
                    "opponent ELO is the game-weighted average (300)",
                    f"opponent ELO {avg_elo}")
    whites = sum(1 for i in range(20) if i % 2 == 0)
    blacks = 20 - whites
    all_ok &= check(whites == blacks == 10,
                    "each 20-game block is 10 white / 10 black",
                    f"color split {whites}W/{blacks}B")

    print("\n[legacy baseline adapter]")
    class TinyLegacyConfig:
        MODEL_TYPE = "transformer"
        HEAD_KIND = "value_policy"
        NUM_ACTIONS = 26
        EMBED_VOCAB_SIZE = 31
        CONTEXT_SIZE = 4
        D_MODEL = 32
        DROPOUT = 0.0
        VALUE_HIDDEN = 16
        MAX_SEQ_LEN = 29
        N_HEAD = 4
        N_LAYERS = 1
        DIM_FEEDFORWARD = 64

    legacy = LegacyValueTransformer(config=TinyLegacyConfig()).eval()
    board = torch.zeros(2, 28, dtype=torch.long)
    ctx5 = torch.tensor([
        [0.0, 2.0 / Config.MAX_CUBE, 0.1, 0.2, 0.0],
        [1.0, 4.0 / Config.MAX_CUBE, 0.3, 0.4, 1.0],
    ])
    outcome, cube = legacy(board, ctx5)
    all_ok &= check(tuple(outcome.shape) == (2, 6) and tuple(cube.shape) == (2, 2),
                    "legacy net maps 5-feature context to (outcome, cube)",
                    f"unexpected legacy shapes {tuple(outcome.shape)} / {tuple(cube.shape)}")
    probs = torch.softmax(outcome, dim=-1)
    all_ok &= check(torch.all(probs[:, 1] < 1e-6) and torch.all(probs[:, 2] < 1e-6),
                    "legacy adapter puts no mass on gammon outcomes",
                    f"gammon mass {probs[:, 1:3].tolist()}")

    baseline_path, baseline_config, _ = baseline_artifact_paths()
    if os.path.exists(baseline_path) and os.path.exists(baseline_config):
        loaded, elo = load_model_with_config(baseline_config, baseline_path, "cpu")
        out, cube = loaded(board[:1], ctx5[:1])
        all_ok &= check(tuple(out.shape) == (1, 6) and elo > 0,
                        f"frozen baseline loads (ELO {elo:.0f})",
                        f"baseline forward failed: {tuple(out.shape)} elo={elo}")
    else:
        print("⏭️  skipped real baseline load (checkpoints/baseline missing)")

    print("\n[checkpoint size]")
    max_bytes = 15 * 1024 * 1024
    size_model = get_model()
    optimizer = torch.optim.AdamW(size_model.parameters(), lr=Config.LR)
    for param in size_model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()
    fd, ckpt_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        save_checkpoint(size_model, optimizer, 0, 0.0, 0.0, ckpt_path)
        ckpt_bytes = os.path.getsize(ckpt_path)
    finally:
        os.remove(ckpt_path)
    ckpt_mb = ckpt_bytes / (1024 * 1024)
    all_ok &= check(
        ckpt_bytes < max_bytes,
        f"stage1/stage2 AdamW checkpoint {ckpt_mb:.2f}MB < 15MB",
        f"checkpoint {ckpt_mb:.2f}MB exceeds 15MB",
    )

    print("\n[10] Opening roll, cube price, collection, eval cap")
    opener = BackgammonGame(train_mode=False)
    for _ in range(30):
        rolled = opener.roll_opening()
        higher = 1 if rolled[0] > rolled[1] else -1
        if rolled[0] == rolled[1] or opener.turn != higher or not opener.must_play_dice():
            all_ok &= check(False, "", f"opening roll {rolled} turn {opener.turn}")
            break
    else:
        all_ok &= check(True, "higher die opens and must play that roll", "opening roll")

    _, plain = opener.get_vector(0, 0)
    opener.cube_offered = True
    _, offered = opener.get_vector(0, 0)
    all_ok &= check(
        float(plain[0]) != Config.CUBE_OFFERED and float(offered[0]) == Config.CUBE_OFFERED,
        "a take is encoded differently from an on-roll double",
        f"owner features {float(plain[0])} vs {float(offered[0])}",
    )
    opener.fast_restore(opener.fast_save())
    all_ok &= check(opener.cube_offered, "cube-offered survives save/restore", "snapshot dropped the flag")
    all_ok &= check(model.embedding.padding_idx is None,
                    "15 opposing checkers is a trainable token",
                    "embedding still treats token 0 as padding")

    old_target, old_train = Config.MATCH_TARGET, Config.TRAIN_MODE
    Config.MATCH_TARGET, Config.TRAIN_MODE = 7, False
    try:
        table7 = MatchEquityTable(match_target=7)
        lose_bg = torch.zeros(6)
        lose_bg[5] = 1.0
        gain, _ = cube_decision_gain(lose_bg, 0, 0, 1, table7, is_take=False)
        now = match_equity_from_outcomes(lose_bg, 0, 0, 1, table7)
        doubled = match_equity_from_outcomes(lose_bg, 0, 0, 2, table7)
        all_ok &= check(
            gain < 0 and abs(gain - (doubled - now)) < 1e-6,
            f"certain backgammon loss is a bad double ({gain:+.3f})",
            f"cube price {gain:+.3f}, doubled-now {doubled - now:+.3f}",
        )
    finally:
        Config.MATCH_TARGET, Config.TRAIN_MODE = old_target, old_train

    all_ok &= check(split_matches(4, 16) == [1, 1, 1, 1] and sum(split_matches(10, 4)) == 10,
                    "workers play exactly the requested number of matches",
                    f"split 4/16={split_matches(4, 16)} 10/4={split_matches(10, 4)}")

    class _Reset:
        def reset(self):
            pass

    capped_game = BackgammonGame(train_mode=False)
    cap_winner, cap_points = eval_play_game(
        capped_game, None, None, _Reset(), _Reset(), True, "cpu", 0, 0, max_moves=0
    )
    all_ok &= check(cap_winner in (1, -1) and cap_points == 1,
                    "eval move cap names a one-point winner",
                    f"eval cap returned {(cap_winner, cap_points)}")

    print("\n" + "=" * 40 + ("\n🏁 ALL CHECKS PASSED" if all_ok else "\n🏁 SOME CHECKS FAILED"))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_regression_suite() else 1)
