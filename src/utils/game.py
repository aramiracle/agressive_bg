import random
import torch
from src.config import Config
from src.utils.cube import get_learned_cube_decision
from src.utils.outcome import (
    flip, terminal_distribution, race_distribution, race_win_probability,
)
from src.search import Searcher, select_candidate


def _single_win_points(game):
    mult = Config.R_WIN if game.train_mode else 1
    return game.cube * mult


def _winner_by_pips(game):
    """Break a move-cap deadlock: fewer pips wins, on-roll wins a pip tie."""
    pips_p1 = game.pip_count(1)
    pips_p2 = game.pip_count(-1)
    if pips_p1 == pips_p2:
        return game.turn
    return 1 if pips_p1 < pips_p2 else -1


def _play_single_game(
    game,
    model_p1, searcher_p1,
    model_p2, searcher_p2,
    scores,
    crawford_active,
    device,
    is_eval=False,
    cube_epsilon=0.0,
    equity_table=None,
    equity_table_p2=None,
):
    """
    Play one game. Returns (winner, points, history, stats, terminal_soft).

    history entries (one per decision) hold the position from the deciding
    player's perspective plus the search's outcome estimate of the chosen play
    (used as the TD bootstrap). terminal_soft is None unless the game was cut
    short as a gammon-free race, in which case it maps player -> outcome
    distribution from the analytic race formula.
    """
    game.reset()
    game.set_match_scores(scores[1], scores[-1])
    game.crawford_active = crawford_active
    game.roll_opening()

    searcher_p1.reset()
    searcher_p2.reset()

    history = []
    stats = {
        'doubles': 0, 'takes': 0, 'drops': 0,
        'sum_val_double': 0.0, 'sum_val_drop': 0.0
    }

    for turn_idx in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0:
            break

        current_model    = model_p1    if game.turn == 1 else model_p2
        current_searcher = searcher_p1 if game.turn == 1 else searcher_p2

        active_equity = (
            equity_table if game.turn == 1
            else (equity_table_p2 if equity_table_p2 is not None else equity_table)
        )

        my_s  = scores[game.turn]
        opp_s = scores[-game.turn]

        # ==========================================
        # 1. CUBING PHASE
        # ==========================================
        if game.can_double() and not game.crawford_active:
            double_choice, me_soft_target_d, val_est_doubler, doubler_probs = get_learned_cube_decision(
                current_model, game, device, my_s, opp_s,
                equity_table=active_equity,
                stochastic=not is_eval,
                epsilon=cube_epsilon if not is_eval else 0.0,
                is_take=False
            )

            board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
            if me_soft_target_d is not None:
                cube_probs = me_soft_target_d.cpu()
            else:
                cube_probs = torch.zeros(2)
                cube_probs[double_choice] = 1.0

            history.append({
                'board': board_t, 'ctx': ctx_t,
                'turn': game.turn, 'is_p1': (game.turn == 1),
                'is_cube': True, 'cube_probs': cube_probs,
                'search_probs': None,
            })

            if double_choice == 1:
                stats['doubles'] += 1
                stats['sum_val_double'] += val_est_doubler
                game.switch_turn()
                # The doubler still rolls. Mark the input so the cube head learns
                # take/drop here instead of another on-roll double.
                game.cube_offered = True

                opp_model  = model_p1 if game.turn == 1 else model_p2
                opp_equity = (
                    equity_table if game.turn == 1
                    else (equity_table_p2 if equity_table_p2 is not None else equity_table)
                )

                take_choice, me_soft_target_t, val_est_taker, _ = get_learned_cube_decision(
                    opp_model, game, device, opp_s, my_s,
                    equity_table=opp_equity,
                    stochastic=not is_eval,
                    epsilon=cube_epsilon if not is_eval else 0.0,
                    is_take=True,
                    equity_probs=flip(doubler_probs),
                )

                board_opp, ctx_opp = game.get_vector(opp_s, my_s, device='cpu', canonical=True)
                if me_soft_target_t is not None:
                    take_probs = me_soft_target_t.cpu()
                else:
                    take_probs = torch.zeros(2)
                    take_probs[take_choice] = 1.0

                history.append({
                    'board': board_opp, 'ctx': ctx_opp,
                    'turn': game.turn, 'is_p1': (game.turn == 1),
                    'is_cube': True, 'cube_probs': take_probs,
                    'search_probs': None,
                })

                game.cube_offered = False
                game.switch_turn()  # Switch back to original player

                if take_choice == 1:
                    stats['takes'] += 1
                    game.apply_double()
                else:
                    stats['drops'] += 1
                    stats['sum_val_drop'] += val_est_taker
                    winner, points = game.handle_cube_refusal()
                    return winner, points, history, stats, None

        # ==========================================
        # 2. RACE CUT-OFF (B6) — self-play only
        # ==========================================
        if (not is_eval and Config.RACE_EARLY_TERMINATION and game.is_gammon_free_race()):
            on_roll = game.turn
            p = race_win_probability(game.pip_count(on_roll), game.pip_count(-on_roll))
            terminal_soft = {on_roll: race_distribution(p), -on_roll: race_distribution(1.0 - p)}
            winner = on_roll if random.random() < p else -on_roll
            return winner, _single_win_points(game), history, stats, terminal_soft

        # ==========================================
        # 3. MOVEMENT PHASE — one search per turn
        # ==========================================
        if not game.dice:
            game.roll_dice()
        result = current_searcher.search(
            game, my_s, opp_s, stochastic=(not is_eval)
        )

        if len(result) > 0:
            explore = (not is_eval) and turn_idx < Config.EXPLORE_TURNS
            chosen = select_candidate(result, explore)

            board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
            history.append({
                'board': board_t, 'ctx': ctx_t,
                'turn': game.turn, 'is_p1': (game.turn == 1),
                'is_cube': False, 'cube_probs': None,
                'search_probs': chosen.probs.clone(),
            })

            game.apply_turn(chosen.path)
            if game.check_win()[0] != 0:
                break

        game.switch_turn()

    winner, total_points = game.check_win()
    if winner == 0:
        # MAX_GAME_MOVES hit with nobody borne off (random nets can loop).
        winner = _winner_by_pips(game)
        total_points = _single_win_points(game)
        if game.crawford_active:
            game.crawford_used = True
    return winner, total_points, history, stats, None


# ---------------------------------------------------------------------------
# Target assignment: TD(lambda) over outcome distributions (B4)
# ---------------------------------------------------------------------------
def _assign_targets(game_history, winner, win_type, terminal_soft=None):
    """
    Walk the game backwards, per player, building
        target_t = (1 - lambda) * V_search(s_{t+1}) + lambda * target_{t+1}
    where V_search(s_{t+1}) is the search estimate at that player's next
    decision and the chain terminates in the exact (or analytic race) outcome.

    Cube decisions share the target of the move decision that follows them.

    Returns tuples: (board, ctx, outcome_target[6], is_cube, cube_target[2])
    """
    if terminal_soft is not None:
        terminal = {1: terminal_soft[1], -1: terminal_soft[-1]}
    else:
        terminal = {
            1:  terminal_distribution(winner == 1,  win_type),
            -1: terminal_distribution(winner == -1, win_type),
        }

    lam = Config.TD_LAMBDA
    last_target = dict(terminal)
    last_est    = dict(terminal)
    zeros2      = torch.zeros(2)

    out = []
    for h in reversed(game_history):
        p = h['turn']
        if h['is_cube']:
            target = last_target[p]
            out.append((h['board'], h['ctx'], target, True, h['cube_probs']))
        else:
            target = (1.0 - lam) * last_est[p] + lam * last_target[p]
            last_est[p]    = h['search_probs']
            last_target[p] = target
            out.append((h['board'], h['ctx'], target, False, zeros2))
    out.reverse()
    return out


def _game_win_type(game, winner, dropped):
    if dropped or winner == 0:
        return 1
    return game.win_type(winner)


def play_self_play_match(game, searcher, model, device, match_equity_table,
                         is_eval=False, cube_epsilon=0.0):
    scores             = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred  = False
    scores_seen_p1     = []
    scores_seen_p2     = []

    match_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                   'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if (not crawford_occurred and Config.MATCH_TARGET > 1
                and (Config.MATCH_TARGET - scores[leader] == 1)):
            is_crawford_game  = True
            crawford_occurred = True

        scores_seen_p1.append((scores[1],  scores[-1]))
        scores_seen_p2.append((scores[-1], scores[1]))

        winner, points, game_history, g_stats, terminal_soft = _play_single_game(
            game, model, searcher, model, searcher, scores, is_crawford_game, device,
            is_eval=is_eval, cube_epsilon=cube_epsilon,
            equity_table=match_equity_table,
        )
        dropped = game.check_win()[0] == 0 and terminal_soft is None
        win_type = _game_win_type(game, winner, dropped)

        scores[winner] += points
        match_stats['games'] += 1
        for k in g_stats: match_stats[k] += g_stats[k]

        full_match_history.extend(
            _assign_targets(game_history, winner, win_type, terminal_soft)
        )

    overall_winner = 1 if scores[1] >= Config.MATCH_TARGET else -1
    match_equity_table.update_from_match(scores_seen_p1, i_won=(overall_winner == 1))
    match_equity_table.update_from_match(scores_seen_p2, i_won=(overall_winner == -1))

    return full_match_history, overall_winner, match_stats


def play_vs_baseline_match(game, current_model, baseline_model, searcher_current, device,
                           match_equity_table, cube_epsilon=0.0,
                           baseline_equity_table=None):
    scores             = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred  = False
    scores_seen_p1     = []
    scores_seen_p2     = []

    match_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                   'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    current_is_p1 = random.choice([True, False])
    et_current  = match_equity_table
    et_baseline = baseline_equity_table if baseline_equity_table is not None else match_equity_table
    searcher_baseline = Searcher(baseline_model, device=device, equity_table=et_baseline)

    model_p1    = current_model     if current_is_p1 else baseline_model
    model_p2    = baseline_model    if current_is_p1 else current_model
    searcher_p1 = searcher_current  if current_is_p1 else searcher_baseline
    searcher_p2 = searcher_baseline if current_is_p1 else searcher_current
    et_p1 = et_current  if current_is_p1 else et_baseline
    et_p2 = et_baseline if current_is_p1 else et_current

    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if (not crawford_occurred and Config.MATCH_TARGET > 1
                and (Config.MATCH_TARGET - scores[leader] == 1)):
            is_crawford_game  = True
            crawford_occurred = True

        scores_seen_p1.append((scores[1],  scores[-1]))
        scores_seen_p2.append((scores[-1], scores[1]))

        winner, points, game_history, g_stats, terminal_soft = _play_single_game(
            game, model_p1, searcher_p1, model_p2, searcher_p2, scores, is_crawford_game, device,
            is_eval=False, cube_epsilon=cube_epsilon,
            equity_table=et_p1, equity_table_p2=et_p2,
        )
        dropped = game.check_win()[0] == 0 and terminal_soft is None
        win_type = _game_win_type(game, winner, dropped)

        scores[winner] += points
        match_stats['games'] += 1
        for k in g_stats: match_stats[k] += g_stats[k]

        # TD(lambda) must walk the full game so bootstraps stay on-policy;
        # only the current model's own decisions are stored as training data.
        assigned = _assign_targets(game_history, winner, win_type, terminal_soft)
        full_match_history.extend(
            t for t, h in zip(assigned, game_history) if h['is_p1'] == current_is_p1
        )

    overall_winner = 1 if scores[1] >= Config.MATCH_TARGET else -1
    match_equity_table.update_from_match(scores_seen_p1, i_won=(overall_winner == 1))
    match_equity_table.update_from_match(scores_seen_p2, i_won=(overall_winner == -1))

    current_won_match = (scores[1] >= Config.MATCH_TARGET) if current_is_p1 else \
                        (scores[-1] >= Config.MATCH_TARGET)
    return full_match_history, current_won_match, match_stats
