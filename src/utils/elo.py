"""ELO rating system for model evaluation with Learned Cubing."""

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from src.config import Config
from src.mcts import MCTS
from src.engine import BackgammonGame

torch.multiprocessing.set_sharing_strategy("file_system")


def calculate_expected_score(player_elo, opponent_elo):
    """Calculate expected score using standard ELO formula."""
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / Config.ELO_SCALE))


def update_elo(current_elo, opponent_elo, wins, total_games):
    """Update ELO based on match results."""
    if total_games == 0:
        return current_elo

    actual = wins / total_games
    expected = calculate_expected_score(current_elo, opponent_elo)
    delta = Config.ELO_K * (actual - expected)

    delta = max(-Config.ELO_SCALE, min(Config.ELO_SCALE, delta))
    return current_elo + delta * total_games


def get_cube_action(model, game, device, my_score=0, opp_score=0):
    """Consult the model's learned cube policy."""
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        _, _, _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        action = torch.argmax(cube_logits.squeeze(0)).item()
    return action


def play_single_game(game, model_a, model_b, mcts_a, mcts_b, a_is_white, device,
                     score_a, score_b, max_moves=1000):
    """Plays a single game within a match context."""
    game.reset()
    mcts_a.reset()
    mcts_b.reset()

    move_count = 0
    while move_count < max_moves:
        winner, points = game.check_win()
        if winner != 0: break

        is_a_turn = (game.turn == 1 and a_is_white) or (game.turn == -1 and not a_is_white)
        p1_score, p2_score = (score_a, score_b) if is_a_turn else (score_b, score_a)

        active_model = model_a if is_a_turn else model_b
        opp_model    = model_b if is_a_turn else model_a
        active_mcts  = mcts_a  if is_a_turn else mcts_b

        # ---------------- 1. Learned Doubling ----------------
        if game.can_double():
            if get_cube_action(active_model, game, device, p1_score, p2_score) == 1:
                game.switch_turn()
                take_decision = get_cube_action(opp_model, game, device, p2_score, p1_score)
                game.switch_turn()

                if take_decision == 1:
                    game.apply_double()
                else:
                    win_side, cube_val = game.handle_cube_refusal()
                    return win_side, cube_val

        # ---------------- 2. Movement ----------------
        game.roll_dice()
        while game.dice:
            legal = game.get_legal_moves()
            if not legal: break

            root   = active_mcts.search(game, p1_score, p2_score)
            action = max(root.children, key=lambda n: n.visits).action

            game.step_atomic(action)
            active_mcts.advance_to_child(action)
            move_count += 1
            if game.check_win()[0] != 0: break

        if game.check_win()[0] == 0:
            game.switch_turn()

    winner, points = game.check_win()
    return winner, points * game.cube


def _worker_play_match(args):
    match_idx, model_a_state, model_b_state, device = args

    torch.set_num_threads(1)

    # Recreate models from state dicts inside the worker (spawn-safe)
    from src.model import get_model
    model_a = get_model().to(device)
    model_a.load_state_dict(model_a_state)
    model_a.eval()

    model_b = get_model().to(device)
    model_b.load_state_dict(model_b_state)
    model_b.eval()

    game_instance = BackgammonGame()
    mcts_a = MCTS(model_a, device=device)
    mcts_b = MCTS(model_b, device=device)

    score_a, score_b = 0, 0
    target = Config.MATCH_TARGET
    a_is_white = (match_idx % 2 == 0)

    while score_a < target and score_b < target:
        winner, points = play_single_game(
            game_instance,
            model_a, model_b,
            mcts_a, mcts_b,
            a_is_white, device,
            score_a, score_b
        )

        if (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
            score_a += points
        else:
            score_b += points

    return 1.0 if score_a >= target else 0.0


def evaluate_vs_opponent(args):
    """
    Play num_games matches of model_a vs model_b.
    Returns (wins_by_model_a, num_games).
    """
    game, model_a, model_b, num_games, device, num_processes = args

    if num_processes is None:
        num_processes = mp.cpu_count()

    model_a_state = model_a.state_dict()
    model_b_state = model_b.state_dict()

    ctx = mp.get_context("spawn")

    wins = 0.0
    pbar = tqdm(
        total=num_games,
        desc=f"ELO eval ({num_processes} cores)",
        dynamic_ncols=True,
        leave=False
    )

    worker_args = [
        (i, model_a_state, model_b_state, device)
        for i in range(num_games)
    ]

    with ctx.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(_worker_play_match, worker_args):
            wins += result
            pbar.update(1)
            pbar.set_postfix({"wins": f"{int(wins)}/{pbar.n}"})

    pbar.close()
    return wins, num_games


def evaluate_combined(model, best_model, baseline_model,
                      best_elo, baseline_elo,
                      total_games, device, num_processes=None):
    """
    Split eval games between baseline and best model according to
    Config.BASELINE_SELF_PLAY_RATIO, accumulate results, and compute
    the weighted opponent ELO.

    Split rule (matching the spec example):
        BASELINE_SELF_PLAY_RATIO = 0.7, ELO_EVAL_GAMES = 100
        n_vs_baseline = round(100 * 0.7) = 70   <- vs external baseline
        n_vs_best     = 100 - 70          = 30   <- vs best self-play model
        opponent_elo  = 1000*0.7 + 500*0.3 = 850

    When baseline_model is None (pure self-play trainer), all games are
    played against best_model and opponent_elo = best_elo.

    Returns:
        total_wins    (float)  – accumulated wins across all games
        total_games   (int)    – == total_games argument
        opponent_elo  (float)  – weighted ELO of the mixed opponent pool
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    ratio = Config.BASELINE_SELF_PLAY_RATIO  # fraction of eval games vs baseline

    if baseline_model is None:
        # No external baseline available – fall back to best model only
        n_vs_baseline = 0
        n_vs_best     = total_games
    else:
        n_vs_baseline = round(total_games * ratio)
        n_vs_best     = total_games - n_vs_baseline

    wins_vs_baseline = 0.0
    wins_vs_best     = 0.0

    # --- Games vs baseline ---
    if n_vs_baseline > 0:
        tqdm.write(
            f"   ELO eval: {n_vs_baseline} games vs baseline (ELO {baseline_elo:.0f})"
        )
        wins_vs_baseline, _ = evaluate_vs_opponent(
            (None, model, baseline_model, n_vs_baseline, device, num_processes)
        )

    # --- Games vs best model ---
    if n_vs_best > 0:
        tqdm.write(
            f"   ELO eval: {n_vs_best} games vs best (ELO {best_elo:.0f})"
        )
        wins_vs_best, _ = evaluate_vs_opponent(
            (None, model, best_model, n_vs_best, device, num_processes)
        )

    total_wins = wins_vs_baseline + wins_vs_best

    # Weighted opponent ELO: baseline contributes its fraction, best contributes remainder
    if n_vs_baseline == 0:
        opponent_elo = best_elo
    elif n_vs_best == 0:
        opponent_elo = baseline_elo
    else:
        w_base = n_vs_baseline / total_games   # == ratio
        w_best = n_vs_best     / total_games   # == 1 - ratio
        opponent_elo = baseline_elo * w_base + best_elo * w_best

    tqdm.write(
        f"   ELO eval total: {int(total_wins)}/{total_games} wins "
        f"| opponent_elo={opponent_elo:.1f} "
        f"(baseline×{n_vs_baseline} + best×{n_vs_best})"
    )

    return total_wins, total_games, opponent_elo