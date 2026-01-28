
import random
import torch
from src.backgammon.config import Config
from src.backgammon.utils.cube import get_learned_cube_decision
from src.backgammon.utils.history import finalize_history
from src.backgammon.utils.move import move_to_indices
from src.backgammon.mcts import MCTS

def _play_single_game(
    game, 
    model_p1, mcts_p1, 
    model_p2, mcts_p2, 
    scores, 
    crawford_active, 
    device, 
    collect_history_p1=True, 
    collect_history_p2=True,
    is_eval=False
):
    """
    Plays a single game within a match.
    Returns: (winner_color, points, history_list)
    """
    game.reset()
    game.set_match_scores(scores[1], scores[-1])
    game.crawford_active = crawford_active
    
    # Reset MCTS trees
    mcts_p1.reset()
    mcts_p2.reset()
    
    history = []

    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0: break

        # Determine current player's objects
        current_model = model_p1 if game.turn == 1 else model_p2
        current_mcts = mcts_p1 if game.turn == 1 else mcts_p2
        collect_current = collect_history_p1 if game.turn == 1 else collect_history_p2
        
        my_s = scores[game.turn]
        opp_s = scores[-game.turn]
        
        # 1. CUBING PHASE
        if game.can_double() and not game.crawford_active:
            # Decide to double?
            double_choice, double_probs = get_learned_cube_decision(
                current_model, game, device, my_s, opp_s, stochastic=not is_eval
            )
            
            if collect_current:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                history.append({
                    'board': board_t, 'ctx': ctx_t, 'action': double_choice, 
                    'turn': game.turn, 'is_cube': True, 'probs': double_probs,
                    'is_p1': (game.turn == 1)
                })

            if double_choice == 1: # Double offered
                game.switch_turn()
                
                # Opponent decides to take/drop
                opp_model = model_p1 if game.turn == 1 else model_p2
                collect_opp = collect_history_p1 if game.turn == 1 else collect_history_p2
                
                take_choice, take_probs = get_learned_cube_decision(
                    opp_model, game, device, opp_s, my_s, stochastic=not is_eval
                )
                
                if collect_opp:
                    board_opp, ctx_opp = game.get_vector(opp_s, my_s, device='cpu', canonical=True)
                    history.append({
                        'board': board_opp, 'ctx': ctx_opp, 'action': take_choice,
                        'turn': game.turn, 'is_cube': True, 'probs': take_probs,
                        'is_p1': (game.turn == 1)
                    })

                game.switch_turn() # Switch back to original player

                if take_choice == 1:
                    game.apply_double()
                else:
                    # Drop: Original player wins
                    winner, points = game.handle_cube_refusal()
                    return winner, points, history

        # 2. MOVEMENT PHASE
        game.roll_dice()
        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                game.dice = []
                break

            root = current_mcts.search(game, my_s, opp_s)
            children = root.children
            
            # Policy Probability from Visits
            visits = torch.tensor([c.visits for c in children], dtype=torch.float)
            if visits.sum() > 0:
                probs = visits / visits.sum()
                idx = torch.multinomial(probs, 1).item()
            else:
                probs = torch.ones(len(children)) / len(children)
                idx = 0
            
            # Select action
            chosen_action = children[idx].action if children else random.choice(legal)

            # Store Policy Targets
            if collect_current:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                target_f, target_t = torch.zeros(26), torch.zeros(26)
                
                for i, child in enumerate(children):
                    c_act = game.real_action_to_canonical(child.action)
                    s_idx, e_idx = move_to_indices(c_act[0], c_act[1])
                    if 0 <= s_idx < 26: target_f[s_idx] += probs[i]
                    if 0 <= e_idx < 26: target_t[e_idx] += probs[i]
                
                history.append({
                    'board': board_t, 'ctx': ctx_t, 'action': None,
                    'turn': game.turn, 'is_cube': False, 'probs': (target_f, target_t),
                    'is_p1': (game.turn == 1)
                })

            game.step_atomic(chosen_action)
            current_mcts.advance_to_child(chosen_action)
            if game.check_win()[0] != 0: break
        
        if game.check_win()[0] == 0:
            game.switch_turn()
            # MCTS needs reset or tree trimming at turn change? 
            # Simple approach: reset trees on turn change to free memory and prevent stale paths
            mcts_p1.reset()
            mcts_p2.reset()

    winner, total_points = game.check_win()
    return winner, total_points * game.cube, history


# ---------------------------------------------------------
# Match Functions
# ---------------------------------------------------------

def play_self_play_match(game, mcts, model, device, is_eval=False):
    """
    Plays a full match (Self-Play) until Config.MATCH_TARGET is reached.
    """
    scores = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred = False
    
    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        # Determine Crawford status
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if not crawford_occurred and (Config.MATCH_TARGET - scores[leader] == 1):
            is_crawford_game = True
            crawford_occurred = True

        # Play one game
        winner, points, game_history = _play_single_game(
            game=game,
            model_p1=model, mcts_p1=mcts,
            model_p2=model, mcts_p2=mcts,
            scores=scores,
            crawford_active=is_crawford_game,
            device=device,
            collect_history_p1=True,
            collect_history_p2=True,
            is_eval=is_eval
        )

        scores[winner] += points
        
        # Finalize history for this specific game
        # We transform the dictionary structure back to the tuple structure expected by training
        raw_history_tuples = []
        for h in game_history:
            # In self play, we collect everything. 'current_won' depends on whose turn it was vs winner
            player_won_this_game = (winner == 1) if h['is_p1'] else (winner == -1)
            raw_history_tuples.append((
                h['board'], h['ctx'], h['action'], h['turn'], h['is_cube'], h['probs']
            ))
            
        finalized_game_data = finalize_history(raw_history_tuples, (winner==1), points) # Note: finalize handles win/loss flip
        
        # Correct logic: finalize_history takes 'current_won' boolean.
        # But our list has mixed P1 and P2 moves. 
        # Better: Process P1 moves and P2 moves separately if we want perfect reward assignment.
        # The existing finalize_history assumes the list is from the perspective of the winner or handled internally.
        # Let's use a specific loop here to be safe:
        
        reward_mag = (float(points) + Config.MATCH_TARGET) / Config.MATCH_TARGET
        
        for h in game_history:
            # Calculate reward relative to the player who made the move
            player_val = 1 if h['is_p1'] else -1
            is_winner = (player_val == winner)
            reward = reward_mag if is_winner else -reward_mag
            
            full_match_history.append((
                h['board'], h['ctx'], h['action'], reward, h['is_cube'], h['probs']
            ))

    overall_winner = 1 if scores[1] >= Config.MATCH_TARGET else -1
    return full_match_history, overall_winner


def play_vs_baseline_match(game, current_model, baseline_model, mcts_current, device):
    """
    Plays a full match: Current Model vs Baseline.
    Only collects training data for the Current Model.
    """
    scores = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred = False
    
    # Randomly assign seats for the match
    current_is_p1 = random.choice([True, False])
    
    # Setup models/MCTS based on seats
    model_p1 = current_model if current_is_p1 else baseline_model
    model_p2 = baseline_model if current_is_p1 else current_model
    # Corrected calls in utils.py
    mcts_p1 = mcts_current if current_is_p1 else MCTS(baseline_model, device=device)
    mcts_p2 = MCTS(baseline_model, device=device) if current_is_p1 else mcts_current

    while scores[1] < Config.MATCH_TARGET and scores[-1] < Config.MATCH_TARGET:
        leader = 1 if scores[1] > scores[-1] else -1
        is_crawford_game = False
        if not crawford_occurred and (Config.MATCH_TARGET - scores[leader] == 1):
            is_crawford_game = True
            crawford_occurred = True

        winner, points, game_history = _play_single_game(
            game=game,
            model_p1=model_p1, mcts_p1=mcts_p1,
            model_p2=model_p2, mcts_p2=mcts_p2,
            scores=scores,
            crawford_active=is_crawford_game,
            device=device,
            collect_history_p1=current_is_p1,      # Only collect if current model is P1
            collect_history_p2=(not current_is_p1), # Only collect if current model is P2
            is_eval=True # Usually vs baseline involves some exploration, but prompt implies eval-like comparison
        )

        scores[winner] += points

        # Calculate Reward
        reward_mag = (float(points) + Config.MATCH_TARGET) / Config.MATCH_TARGET
        
        # Add to history (game_history only contains current_model's moves due to flags above)
        for h in game_history:
            # Did the current model win?
            # If current is P1 and Winner is 1 -> Won
            # If current is P2 and Winner is -1 -> Won
            am_i_p1 = current_is_p1
            i_won = (winner == 1 and am_i_p1) or (winner == -1 and not am_i_p1)
            
            reward = reward_mag if i_won else -reward_mag
            
            full_match_history.append((
                h['board'], h['ctx'], h['action'], reward, h['is_cube'], h['probs']
            ))

    current_won_match = (scores[1] >= Config.MATCH_TARGET) if current_is_p1 else (scores[-1] >= Config.MATCH_TARGET)
    return full_match_history, current_won_match