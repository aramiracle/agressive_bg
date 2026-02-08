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
    cube_epsilon=0.0
):
    game.reset()
    game.set_match_scores(scores[1], scores[-1])
    game.crawford_active = crawford_active
    
    mcts_p1.reset()
    mcts_p2.reset()
    
    history = []
    
    # Stats container for diagnostics
    stats = {
        'doubles': 0, 'takes': 0, 'drops': 0,
        'sum_val_double': 0.0, 'sum_val_drop': 0.0
    }

    # Normalization factor for Match Equity
    norm_factor = float(Config.MATCH_TARGET * (Config.R_BACKGAMMON + 1))

    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0: break

        current_model = model_p1 if game.turn == 1 else model_p2
        current_mcts = mcts_p1 if game.turn == 1 else mcts_p2
        collect_current = collect_history_p1 if game.turn == 1 else collect_history_p2
        
        my_s = scores[game.turn]
        opp_s = scores[-game.turn]
        
        # ==========================================
        # 1. CUBING PHASE
        # ==========================================
        if game.can_double() and not game.crawford_active:
            double_choice, double_probs, val_est_doubler = get_learned_cube_decision(
                current_model, game, device, my_s, opp_s, 
                stochastic=not is_eval, epsilon=cube_epsilon if not is_eval else 0.0
            )
            
            if collect_current:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                history.append({
                    'board': board_t, 'ctx': ctx_t, 'action': double_choice, 
                    'turn': game.turn, 'is_cube': True, 'probs': double_probs,
                    'is_p1': (game.turn == 1), 'cube_type': 'double_decision',
                    'immediate_reward': None,
                    'mcts_val': val_est_doubler # Use network value for cube stats
                })

            if double_choice == 1:
                stats['doubles'] += 1
                stats['sum_val_double'] += val_est_doubler
                game.switch_turn()
                
                opp_model = model_p1 if game.turn == 1 else model_p2
                collect_opp = collect_history_p1 if game.turn == 1 else collect_history_p2
                
                take_choice, take_probs, val_est_taker = get_learned_cube_decision(
                    opp_model, game, device, opp_s, my_s, 
                    stochastic=not is_eval, epsilon=cube_epsilon if not is_eval else 0.0
                )
                
                if collect_opp:
                    board_opp, ctx_opp = game.get_vector(opp_s, my_s, device='cpu', canonical=True)
                    history.append({
                        'board': board_opp, 'ctx': ctx_opp, 'action': take_choice,
                        'turn': game.turn, 'is_cube': True, 'probs': take_probs,
                        'is_p1': (game.turn == 1), 'cube_type': 'take_decision',
                        'immediate_reward': None,
                        'mcts_val': val_est_taker
                    })

                game.switch_turn() # Switch back to original player

                if take_choice == 1:
                    stats['takes'] += 1
                    game.apply_double()
                    # Game continues
                else:
                    stats['drops'] += 1
                    stats['sum_val_drop'] += val_est_taker
                    
                    # --- HANDLING DROP REWARD PROPORTIONALLY ---
                    # FIX: Get points directly from handle_cube_refusal first
                    winner, points = game.handle_cube_refusal()
                    
                    # Now we have the correct points value to calculate reward magnitude
                    points_lost = float(points)
                    drop_reward_mag = points_lost / norm_factor
                    
                    if collect_current:
                        for h in reversed(history):
                            if h.get('cube_type') == 'double_decision':
                                h['immediate_reward'] = drop_reward_mag
                                break
                    
                    if collect_opp:
                        for h in reversed(history):
                            if h.get('cube_type') == 'take_decision':
                                h['immediate_reward'] = -drop_reward_mag
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

            root = current_mcts.search(game, my_s, opp_s)
            children = root.children
            
            # --- MCTS VALUE EXTRACTION ---
            mcts_root_value = root.value() 
            
            visits = torch.tensor([c.visits for c in children], dtype=torch.float)
            if visits.sum() > 0:
                probs = visits / visits.sum()
                idx = torch.multinomial(probs, 1).item()
            else:
                probs = torch.ones(len(children)) / len(children)
                idx = 0
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
                    'immediate_reward': None,
                    'mcts_val': mcts_root_value # Store for mixing
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


def play_self_play_match(game, mcts, model, device, is_eval=False, cube_epsilon=0.0):
    scores = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred = False
    
    match_stats = {'doubles': 0, 'takes': 0, 'drops': 0, 'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}
    norm_factor = float(Config.MATCH_TARGET * (Config.R_BACKGAMMON + 1))

    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if not crawford_occurred and (Config.MATCH_TARGET - scores[leader] == 1):
            is_crawford_game = True
            crawford_occurred = True

        winner, points, game_history, g_stats = _play_single_game(
            game, model, mcts, model, mcts, scores, is_crawford_game, device,
            collect_history_p1=True, collect_history_p2=True,
            is_eval=is_eval, cube_epsilon=cube_epsilon
        )

        scores[winner] += points
        match_stats['games'] += 1
        for k in g_stats: match_stats[k] += g_stats[k]

        # Base reward from outcome
        reward_mag = float(points) / norm_factor
        
        for h in game_history:
            player_val = 1 if h['is_p1'] else -1
            is_winner = (player_val == winner)
            
            # 1. Determine the Game Outcome Reward
            if h.get('immediate_reward') is not None:
                outcome_reward = h['immediate_reward']
            else:
                outcome_reward = reward_mag if is_winner else -reward_mag
            
            # 2. Mix with MCTS Value for moves
            final_target = outcome_reward
            if not h['is_cube'] and h.get('mcts_val') is not None:
                final_target = 0.5 * outcome_reward + 0.5 * h['mcts_val']

            full_match_history.append((
                h['board'], h['ctx'], h['action'], final_target, h['is_cube'], h['probs']
            ))

    overall_winner = 1 if scores[1] >= Config.MATCH_TARGET else -1
    return full_match_history, overall_winner, match_stats


def play_vs_baseline_match(game, current_model, baseline_model, mcts_current, device, cube_epsilon=0.0):
    scores = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred = False
    match_stats = {'doubles': 0, 'takes': 0, 'drops': 0, 'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}
    norm_factor = float(Config.MATCH_TARGET * (Config.R_BACKGAMMON + 1))
    
    current_is_p1 = random.choice([True, False])
    model_p1 = current_model if current_is_p1 else baseline_model
    model_p2 = baseline_model if current_is_p1 else current_model
    mcts_p1 = mcts_current if current_is_p1 else MCTS(baseline_model, device=device, cpuct=Config.C_PUCT, num_sims=Config.NUM_SIMULATIONS)
    mcts_p2 = MCTS(baseline_model, device=device, cpuct=Config.C_PUCT, num_sims=Config.NUM_SIMULATIONS) if current_is_p1 else mcts_current

    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if not crawford_occurred and (Config.MATCH_TARGET - scores[leader] == 1):
            is_crawford_game = True
            crawford_occurred = True

        winner, points, game_history, g_stats = _play_single_game(
            game, model_p1, mcts_p1, model_p2, mcts_p2, scores, is_crawford_game, device,
            collect_history_p1=current_is_p1, collect_history_p2=(not current_is_p1),
            is_eval=False, cube_epsilon=cube_epsilon
        )

        scores[winner] += points
        match_stats['games'] += 1
        for k in g_stats: match_stats[k] += g_stats[k]

        reward_mag = float(points) / norm_factor
        
        for h in game_history:
            am_i_p1 = current_is_p1
            i_won = (winner == 1 and am_i_p1) or (winner == -1 and not am_i_p1)
            
            if h.get('immediate_reward') is not None:
                outcome_reward = h['immediate_reward'] if i_won else -h['immediate_reward']
            else:
                outcome_reward = reward_mag if i_won else -reward_mag
            
            final_target = outcome_reward
            if not h['is_cube'] and h.get('mcts_val') is not None:
                final_target = 0.5 * outcome_reward + 0.5 * h['mcts_val']

            full_match_history.append((
                h['board'], h['ctx'], h['action'], final_target, h['is_cube'], h['probs']
            ))

    current_won_match = (scores[1] >= Config.MATCH_TARGET) if current_is_p1 else (scores[-1] >= Config.MATCH_TARGET)
    return full_match_history, current_won_match, match_stats