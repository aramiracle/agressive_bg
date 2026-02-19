import random
import torch
from src.config import Config
from src.utils.cube import get_learned_cube_decision
from src.mcts import MCTS

def _play_single_game(
    game,
    model_p1, mcts_p1,
    model_p2, mcts_p2,
    scores,
    crawford_active,
    device,
    collect_history_p1=True,
    collect_history_p2=True,
    is_eval=False,
    cube_epsilon=0.0,
    equity_table=None,     # NEW: passed through for ME-aware cube features
):
    game.reset()
    game.set_match_scores(scores[1], scores[-1])
    game.crawford_active = crawford_active

    mcts_p1.reset()
    mcts_p2.reset()

    history = []

    stats = {
        'doubles': 0, 'takes': 0, 'drops': 0,
        'sum_val_double': 0.0, 'sum_val_drop': 0.0
    }

    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0: break

        current_model = model_p1 if game.turn == 1 else model_p2
        current_mcts  = mcts_p1  if game.turn == 1 else mcts_p2
        collect_current = collect_history_p1 if game.turn == 1 else collect_history_p2

        my_s  = scores[game.turn]
        opp_s = scores[-game.turn]

        # ==========================================
        # 1. CUBING PHASE
        # ==========================================
        if game.can_double() and not game.crawford_active:
            # equity_table is forwarded so get_learned_cube_decision can
            # append ME-aware features and return log_prob for REINFORCE.
            double_choice, log_prob_double, val_est_doubler = get_learned_cube_decision(
                current_model, game, device, my_s, opp_s,
                equity_table=equity_table,
                stochastic=not is_eval,
                epsilon=cube_epsilon if not is_eval else 0.0,
            )

            if collect_current:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                # Store one-hot for action recovery in REINFORCE (train.py uses argmax)
                cube_target = torch.zeros(2)
                cube_target[double_choice] = 1.0

                history.append({
                    'board': board_t, 'ctx': ctx_t, 'action': double_choice,
                    'turn': game.turn, 'is_cube': True,
                    'probs': cube_target,
                    'is_p1': (game.turn == 1), 'cube_type': 'double_decision',
                    'score_before': (my_s, opp_s),
                    'immediate_reward': None,
                    'mcts_val': val_est_doubler,
                })

            if double_choice == 1:
                stats['doubles'] += 1
                stats['sum_val_double'] += val_est_doubler
                game.switch_turn()

                opp_model   = model_p1 if game.turn == 1 else model_p2
                collect_opp = collect_history_p1 if game.turn == 1 else collect_history_p2

                take_choice, log_prob_take, val_est_taker = get_learned_cube_decision(
                    opp_model, game, device, opp_s, my_s,
                    equity_table=equity_table,
                    stochastic=not is_eval,
                    epsilon=cube_epsilon if not is_eval else 0.0,
                )

                if collect_opp:
                    board_opp, ctx_opp = game.get_vector(opp_s, my_s, device='cpu', canonical=True)
                    take_target = torch.zeros(2)
                    take_target[take_choice] = 1.0

                    history.append({
                        'board': board_opp, 'ctx': ctx_opp, 'action': take_choice,
                        'turn': game.turn, 'is_cube': True,
                        'probs': take_target,
                        'is_p1': (game.turn == 1), 'cube_type': 'take_decision',
                        'score_before': (opp_s, my_s),
                        'immediate_reward': None,
                        'mcts_val': val_est_taker,
                    })

                game.switch_turn()  # Switch back to original player

                if take_choice == 1:
                    stats['takes'] += 1
                    game.apply_double()
                else:
                    stats['drops'] += 1
                    stats['sum_val_drop'] += val_est_taker

                    winner, points = game.handle_cube_refusal()

                    if collect_current:
                        for h in reversed(history):
                            if h.get('cube_type') == 'double_decision':
                                h['drop_outcome'] = True
                                h['score_after']  = (my_s + points, opp_s)
                                break

                    if collect_opp:
                        for h in reversed(history):
                            if h.get('cube_type') == 'take_decision':
                                h['drop_outcome'] = True
                                h['score_after']  = (opp_s, my_s + points)
                                break

                    return winner, points, history, stats

        # ==========================================
        # 2. MOVEMENT PHASE
        # ==========================================
        game.roll_dice()
        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                game.dice = []
                break

            root    = current_mcts.search(game, my_s, opp_s)
            children = root.children

            mcts_root_value = root.value()

            visits = torch.tensor([c.visits for c in children], dtype=torch.float)
            if visits.sum() > 0:
                probs = visits / visits.sum()
                idx   = torch.multinomial(probs, 1).item()
            else:
                probs = torch.ones(len(children)) / len(children)
                idx   = 0
                mcts_root_value = 0.0

            chosen_action = children[idx].action if children else random.choice(legal)

            if collect_current:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                target_f, target_t = torch.zeros(26), torch.zeros(26)

                for i, child in enumerate(children):
                    c_act = game.real_action_to_canonical(child.action)
                    (src, dst), _ = c_act
                    s_idx = 24 if src == "bar" else src
                    e_idx = 25 if dst == "off" else dst
                    if 0 <= s_idx < 26: target_f[s_idx] += probs[i]
                    if 0 <= e_idx < 26: target_t[e_idx] += probs[i]

                history.append({
                    'board': board_t,
                    'ctx': ctx_t,
                    'action': None,
                    'turn': game.turn,
                    'is_cube': False,
                    'probs': (target_f, target_t),
                    'is_p1': (game.turn == 1),
                    'score_before': (my_s, opp_s),
                    'immediate_reward': None,
                    'mcts_val': mcts_root_value,
                })

            game.step_atomic(chosen_action)
            current_mcts.advance_to_child(chosen_action)
            if game.check_win()[0] != 0: break

        if game.check_win()[0] == 0:
            game.switch_turn()
            mcts_p1.reset()
            mcts_p2.reset()

    winner, total_points = game.check_win()
    return winner, total_points, history, stats


# ---------------------------------------------------------------------------
# Helper: shared reward assignment logic (avoids duplication across match fns)
# ---------------------------------------------------------------------------
def _assign_rewards(game_history, match_equity_table,
                    score_after_p1, score_after_p2):
    """
    Convert raw history entries into replay-buffer tuples.

    Reward = match equity change.  For move positions we mix in the MCTS
    value estimate as a variance-reduction baseline (0.5 / 0.5 blend).

    Returns list of (board, ctx, action, reward, is_cube, probs) tuples.
    """
    result = []
    for h in game_history:
        player_val       = 1 if h['is_p1'] else -1
        my_score_before, opp_score_before = h['score_before']

        if h.get('drop_outcome'):
            my_score_after, opp_score_after = h['score_after']
        else:
            if player_val == 1:
                my_score_after, opp_score_after = score_after_p1, score_after_p2
            else:
                my_score_after, opp_score_after = score_after_p2, score_after_p1

        outcome_reward = match_equity_table.compute_reward(
            my_score_before, opp_score_before,
            my_score_after,  opp_score_after,
        )

        final_target = outcome_reward
        if not h['is_cube'] and h.get('mcts_val') is not None:
            final_target = 0.5 * outcome_reward + 0.5 * h['mcts_val']

        result.append((
            h['board'], h['ctx'], h['action'], final_target, h['is_cube'], h['probs']
        ))
    return result


def play_self_play_match(game, mcts, model, device, match_equity_table,
                         is_eval=False, cube_epsilon=0.0):
    scores            = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred  = False
    scores_seen        = []

    match_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                   'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if not crawford_occurred and (Config.MATCH_TARGET - scores[leader] == 1):
            is_crawford_game  = True
            crawford_occurred = True

        score_before_p1 = scores[1]
        score_before_p2 = scores[-1]
        scores_seen.append((score_before_p1, score_before_p2))
        scores_seen.append((score_before_p2, score_before_p1))

        winner, points, game_history, g_stats = _play_single_game(
            game, model, mcts, model, mcts, scores, is_crawford_game, device,
            collect_history_p1=True, collect_history_p2=True,
            is_eval=is_eval, cube_epsilon=cube_epsilon,
            equity_table=match_equity_table,          # NEW
        )

        scores[winner] += points
        match_stats['games'] += 1
        for k in g_stats: match_stats[k] += g_stats[k]

        full_match_history.extend(
            _assign_rewards(game_history, match_equity_table,
                            scores[1], scores[-1])
        )

    overall_winner = 1 if scores[1] >= Config.MATCH_TARGET else -1
    match_equity_table.update_from_match(scores_seen, overall_winner)

    return full_match_history, overall_winner, match_stats


def play_vs_baseline_match(game, current_model, baseline_model, mcts_current, device,
                           match_equity_table, cube_epsilon=0.0):
    scores             = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred  = False
    scores_seen        = []

    match_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                   'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    current_is_p1 = random.choice([True, False])
    model_p1 = current_model  if current_is_p1 else baseline_model
    model_p2 = baseline_model if current_is_p1 else current_model
    mcts_p1  = mcts_current   if current_is_p1 else MCTS(
        baseline_model, device=device, cpuct=Config.C_PUCT, num_sims=Config.NUM_SIMULATIONS)
    mcts_p2  = MCTS(baseline_model, device=device, cpuct=Config.C_PUCT,
                    num_sims=Config.NUM_SIMULATIONS) if current_is_p1 else mcts_current

    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if not crawford_occurred and (Config.MATCH_TARGET - scores[leader] == 1):
            is_crawford_game  = True
            crawford_occurred = True

        score_before_p1 = scores[1]
        score_before_p2 = scores[-1]
        scores_seen.append((score_before_p1, score_before_p2))
        scores_seen.append((score_before_p2, score_before_p1))

        winner, points, game_history, g_stats = _play_single_game(
            game, model_p1, mcts_p1, model_p2, mcts_p2, scores, is_crawford_game, device,
            collect_history_p1=current_is_p1, collect_history_p2=(not current_is_p1),
            is_eval=False, cube_epsilon=cube_epsilon,
            equity_table=match_equity_table,          # NEW
        )

        scores[winner] += points
        match_stats['games'] += 1
        for k in g_stats: match_stats[k] += g_stats[k]

        score_after_p1 = scores[1]
        score_after_p2 = scores[-1]

        for h in game_history:
            # Remap scores to current-model perspective
            my_score_before, opp_score_before = h['score_before']

            if h.get('drop_outcome'):
                my_score_after, opp_score_after = h['score_after']
            elif h['is_p1'] == current_is_p1:
                my_score_after  = score_after_p1 if current_is_p1 else score_after_p2
                opp_score_after = score_after_p2 if current_is_p1 else score_after_p1
            else:
                my_score_after  = score_after_p2 if current_is_p1 else score_after_p1
                opp_score_after = score_after_p1 if current_is_p1 else score_after_p2

            outcome_reward = match_equity_table.compute_reward(
                my_score_before, opp_score_before,
                my_score_after,  opp_score_after,
            )

            final_target = outcome_reward
            if not h['is_cube'] and h.get('mcts_val') is not None:
                final_target = 0.5 * outcome_reward + 0.5 * h['mcts_val']

            full_match_history.append((
                h['board'], h['ctx'], h['action'], final_target, h['is_cube'], h['probs']
            ))

    overall_winner    = 1 if scores[1] >= Config.MATCH_TARGET else -1
    match_equity_table.update_from_match(scores_seen, overall_winner)

    current_won_match = (scores[1] >= Config.MATCH_TARGET) if current_is_p1 else \
                        (scores[-1] >= Config.MATCH_TARGET)
    return full_match_history, current_won_match, match_stats