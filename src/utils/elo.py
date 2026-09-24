"""ELO rating system for model evaluation with Learned Cubing."""

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from src.config import Config
from src.search import Searcher
from src.engine import BackgammonGame
from src.utils.game import _single_win_points, _winner_by_pips
from src.utils.match_equity import MatchEquityTable

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


def promoted_elo(champion_elo, wins_vs_champion, games_vs_champion):
    """
    Rating to store when a candidate replaces the champion.

    The running rating falls on failed gates and, with K=1, a gate pass
    (just over 53%) cannot close that gap. max(champion, running) then
    saves the new weights under the old rating. After promotion both
    checkpoints are the same weights, and those weights just played
    `games_vs_champion` against the previous champion, so the published
    rating is the champion rating updated by that match.
    """
    return update_elo(champion_elo, champion_elo, wins_vs_champion, games_vs_champion)


def get_cube_action(model, game, device, my_score=0, opp_score=0):
    """Consult the model's learned cube policy."""
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.inference_mode():
        _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        action = torch.argmax(cube_logits.squeeze(0)).item()
    return action


def play_single_game(game, model_a, model_b, searcher_a, searcher_b, a_is_white, device,
                     score_a, score_b, max_moves=1000):
    """Plays a single game within a match context (greedy, no exploration)."""
    game.reset()
    game.set_match_scores(score_a if a_is_white else score_b,
                          score_b if a_is_white else score_a)
    searcher_a.reset()
    searcher_b.reset()
    game.roll_opening()

    move_count = 0
    while move_count < max_moves:
        move_count += 1
        winner, points = game.check_win()
        if winner != 0: break

        is_a_turn = (game.turn == 1 and a_is_white) or (game.turn == -1 and not a_is_white)
        p1_score, p2_score = (score_a, score_b) if is_a_turn else (score_b, score_a)

        active_model    = model_a    if is_a_turn else model_b
        opp_model       = model_b    if is_a_turn else model_a
        active_searcher = searcher_a if is_a_turn else searcher_b

        # ---------------- 1. Learned Doubling ----------------
        if game.can_double():
            if get_cube_action(active_model, game, device, p1_score, p2_score) == 1:
                game.switch_turn()
                game.cube_offered = True
                take_decision = get_cube_action(opp_model, game, device, p2_score, p1_score)
                game.cube_offered = False
                game.switch_turn()

                if take_decision == 1:
                    game.apply_double()
                else:
                    win_side, cube_val = game.handle_cube_refusal()
                    return win_side, cube_val

        # ---------------- 2. Movement ----------------
        if not game.dice:
            game.roll_dice()
        result = active_searcher.search(game, p1_score, p2_score, stochastic=False)
        if len(result) > 0:
            game.apply_turn(result.best().path)
            if game.check_win()[0] != 0: break

        game.switch_turn()

    winner, points = game.check_win()
    if winner == 0:
        winner = _winner_by_pips(game)
        points = _single_win_points(game)
        if game.crawford_active:
            game.crawford_used = True
    return winner, points


def _worker_play_match(args):
    (match_idx, model_a_state, model_b_state, model_b_config_path,
     device, equity_table_state) = args

    torch.set_num_threads(1)

    from src.model import get_model
    from src.utils.checkpoint import build_model_from_config_path

    model_a = get_model().to(device)
    model_a.load_state_dict(model_a_state)
    model_a.eval()

    if model_b_config_path is not None:
        model_b = build_model_from_config_path(model_b_config_path, device)
    else:
        model_b = get_model().to(device)
    model_b.load_state_dict(model_b_state)
    model_b.eval()

    equity_table = None
    if equity_table_state is not None:
        equity_table = MatchEquityTable()
        equity_table.equity_table = equity_table_state

    game_instance = BackgammonGame()
    searcher_a = Searcher(model_a, device=device, equity_table=equity_table)
    searcher_b = Searcher(model_b, device=device, equity_table=equity_table)

    score_a, score_b = 0, 0
    target = Config.MATCH_TARGET
    # Even match_idx starts as white so each opponent block is 50/50 color.
    a_is_white = (match_idx % 2 == 0)

    while score_a < target and score_b < target:
        winner, points = play_single_game(
            game_instance,
            model_a, model_b,
            searcher_a, searcher_b,
            a_is_white, device,
            score_a, score_b
        )

        if (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
            score_a += points
        else:
            score_b += points

        a_is_white = not a_is_white

    return 1.0 if score_a >= target else 0.0


def evaluate_vs_opponent(args):
    """
    Play num_games matches of model_a vs model_b.

    Color: even match_idx starts as white, odd as black, and sides swap
    after every game inside a match so the candidate plays both colours
    equally. Returns (wins_by_model_a, num_games).
    """
    (game, model_a, model_b, num_games, device, num_processes,
     model_b_config_path, equity_table_state) = args

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
        (i, model_a_state, model_b_state, model_b_config_path, device, equity_table_state)
        for i in range(num_games)
    ]

    with ctx.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(_worker_play_match, worker_args):
            wins += result
            pbar.update(1)
            pbar.set_postfix({"wins": f"{int(wins)}/{pbar.n}"})

    pbar.close()
    return wins, num_games


def plays_against_baseline(has_baseline, _current_elo, best_elo, baseline_elo):
    """
    Whether this iteration's training games should face the frozen baseline.

    Eval already splits onto the baseline while best_model is below it.
    Collection follows that published rating only. The running rating lags
    on failed gates and, with K=1, does not catch up on a pass, so it must
    not abandon self-play while the weights are still the champion.

    Returns:
        True when a baseline is loaded and best_model is still under it.
    """
    if not has_baseline:
        return False
    return best_elo < baseline_elo


def split_eval_games(total_games, best_elo, baseline_elo, has_baseline):
    """
    How many GATE_GAMES go to best vs baseline.

    While best_model is weaker than the frozen baseline, split the eval
    set in half so the rating is not only vs a weak clone of itself.
    Once best catches up, every game is vs best_model.

    Returns:
        (n_vs_best, n_vs_baseline)
    """
    if has_baseline and best_elo < baseline_elo:
        n_vs_baseline = total_games // 2
        n_vs_best = total_games - n_vs_baseline
        return n_vs_best, n_vs_baseline
    return total_games, 0


def mixed_opponent_elo(best_elo, baseline_elo, n_vs_best, n_vs_baseline):
    """Average opponent ELO, weighted by games actually played against each."""
    played = n_vs_best + n_vs_baseline
    if played == 0:
        return best_elo
    return (best_elo * n_vs_best + baseline_elo * n_vs_baseline) / played


def evaluate_combined(model, best_model, baseline_model,
                      best_elo, baseline_elo,
                      total_games, device, num_processes=None,
                      baseline_config_path=None, equity_table=None):
    """
    Play `total_games` matches (trainers pass GATE_GAMES).

    When best_elo is below baseline_elo, half the matches are vs the
    baseline and half vs best_model; otherwise all matches are vs best.
    Each block alternates color so the candidate plays white and black
    equally against both opponents. Opponent ELO is the game-weighted
    average. The gate still uses only the games against best_model.

    Returns:
        total_wins    (float)  – accumulated wins across all games
        played        (int)    – games actually played
        opponent_elo  (float)  – weighted average ELO of opponents faced
        wins_vs_best  (float)  – wins in the games against best_model (gating)
        n_vs_best     (int)    – number of games against best_model
    """
    equity_table_state = equity_table.equity_table.copy() if equity_table is not None else None
    if num_processes is None:
        num_processes = mp.cpu_count()

    n_vs_best, n_vs_baseline = split_eval_games(
        total_games, best_elo, baseline_elo, baseline_model is not None,
    )

    wins_vs_baseline = 0.0
    wins_vs_best     = 0.0

    def _block(label, n, elo, wins):
        rate = wins / n if n else 0.0
        tqdm.write(f"  {label:<12} {int(wins):>3}/{n:<3} {rate:>6.1%}   elo {elo:.0f}")

    if n_vs_baseline > 0:
        tqdm.write(f"  vs baseline   {n_vs_baseline} matches   elo {baseline_elo:.0f}")
        wins_vs_baseline, _ = evaluate_vs_opponent(
            (None, model, baseline_model, n_vs_baseline, device, num_processes,
             baseline_config_path, equity_table_state)
        )
        _block("baseline", n_vs_baseline, baseline_elo, wins_vs_baseline)

    if n_vs_best > 0:
        tqdm.write(f"  vs best       {n_vs_best} matches   elo {best_elo:.0f}")
        wins_vs_best, _ = evaluate_vs_opponent(
            (None, model, best_model, n_vs_best, device, num_processes, None, equity_table_state)
        )
        _block("best", n_vs_best, best_elo, wins_vs_best)

    played = n_vs_best + n_vs_baseline
    total_wins = wins_vs_baseline + wins_vs_best
    opponent_elo = mixed_opponent_elo(
        best_elo, baseline_elo, n_vs_best, n_vs_baseline,
    )
    rate = total_wins / played if played else 0.0
    tqdm.write(
        f"  {'total':<12} {int(total_wins):>3}/{played:<3} {rate:>6.1%}   "
        f"opp elo {opponent_elo:.0f}"
    )

    return total_wins, played, opponent_elo, wins_vs_best, n_vs_best


def passes_gate(wins_vs_best, n_vs_best, total_wins, total_games):
    """
    E3 gating: promote only if the candidate's win rate against best_model
    exceeds Config.GATE_WIN_RATE. Falls back to the overall win rate when no
    games against best_model were played.
    """
    if n_vs_best > 0:
        rate = wins_vs_best / n_vs_best
    elif total_games > 0:
        rate = total_wins / total_games
    else:
        return False, 0.0
    return rate > Config.GATE_WIN_RATE, rate