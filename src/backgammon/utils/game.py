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
    is_eval=False,
    cube_epsilon=0.0
):
    """
    Plays a single game within a match with proper cube reward attribution.
    
    Returns: (winner_color, points, history_list)
    
    Key changes:
    - Added cube_epsilon parameter for exploration
    - Cube decisions now track immediate rewards
    - Proper reward attribution for double/take/drop decisions
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
        
        # ==========================================
        # 1. CUBING PHASE WITH IMMEDIATE REWARDS
        # ==========================================
        if game.can_double() and not game.crawford_active:
            # Decide to double?
            double_choice, double_probs = get_learned_cube_decision(
                current_model, game, device, my_s, opp_s, 
                stochastic=not is_eval,
                epsilon=cube_epsilon if not is_eval else 0.0
            )
            
            if collect_current:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                # Store with metadata for later reward assignment
                history.append({
                    'board': board_t, 
                    'ctx': ctx_t, 
                    'action': double_choice, 
                    'turn': game.turn, 
                    'is_cube': True, 
                    'probs': double_probs,
                    'is_p1': (game.turn == 1),
                    'cube_type': 'double_decision',
                    'immediate_reward': None  # Will be set below based on outcome
                })

            if double_choice == 1:  # Double offered
                doubler_turn = game.turn
                doubler_is_p1 = (game.turn == 1)
                game.switch_turn()
                
                # Opponent decides to take/drop
                opp_model = model_p1 if game.turn == 1 else model_p2
                collect_opp = collect_history_p1 if game.turn == 1 else collect_history_p2
                
                take_choice, take_probs = get_learned_cube_decision(
                    opp_model, game, device, opp_s, my_s, 
                    stochastic=not is_eval,
                    epsilon=cube_epsilon if not is_eval else 0.0
                )
                
                if collect_opp:
                    board_opp, ctx_opp = game.get_vector(opp_s, my_s, device='cpu', canonical=True)
                    history.append({
                        'board': board_opp, 
                        'ctx': ctx_opp, 
                        'action': take_choice,
                        'turn': game.turn, 
                        'is_cube': True, 
                        'probs': take_probs,
                        'is_p1': (game.turn == 1),
                        'cube_type': 'take_decision',
                        'immediate_reward': None
                    })

                game.switch_turn()  # Back to original player

                if take_choice == 1:  # Opponent takes the double
                    game.apply_double()
                    # Game continues - rewards will come from final outcome
                    # No immediate rewards set (stays None)
                    
                else:  # Opponent drops - IMMEDIATE RESOLUTION
                    winner, points = game.handle_cube_refusal()
                    
                    # CRITICAL: Assign immediate rewards for cube decisions
                    # The doubler won immediately - this was a GOOD double
                    # The dropper lost immediately - need to assess if drop was correct
                    
                    # Find and reward the doubling decision
                    if collect_current:
                        for h in reversed(history):
                            if h.get('cube_type') == 'double_decision':
                                # Successful double that won immediately
                                h['immediate_reward'] = +1.0
                                break
                    
                    # Find and reward the take/drop decision
                    if collect_opp:
                        for h in reversed(history):
                            if h.get('cube_type') == 'take_decision':
                                # Dropped and lost - the drop decision gets negative reward
                                # Note: This might be the correct decision if position was bad,
                                # but we're using the immediate outcome for training
                                h['immediate_reward'] = -1.0
                                break
                    
                    return winner, points, history

        # ==========================================
        # 2. MOVEMENT PHASE (unchanged from original)
        # ==========================================
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
                    # 1. Convert real engine move to canonical (from player's perspective)
                    c_act = game.real_action_to_canonical(child.action)
                    
                    # 2. UNPACK ATOMIC WRAPPER: c_act is ((src, dst), die)
                    (src, dst), _ = c_act
                    
                    # 3. Map board locations to policy indices (0-25)
                    # "bar" is usually mapped to 24, "off" to 25
                    s_idx = 24 if src == "bar" else src
                    e_idx = 25 if dst == "off" else dst
                    
                    # 4. Aggregate visit probabilities into the target tensors
                    if 0 <= s_idx < 26: 
                        target_f[s_idx] += probs[i]
                    if 0 <= e_idx < 26: 
                        target_t[e_idx] += probs[i]
                
                history.append({
                    'board': board_t, 
                    'ctx': ctx_t, 
                    'action': None,
                    'turn': game.turn, 
                    'is_cube': False, 
                    'probs': (target_f, target_t),
                    'is_p1': (game.turn == 1),
                    'immediate_reward': None
                })

            # Execute the action using the wrapper
            game.step_atomic(chosen_action)
            current_mcts.advance_to_child(chosen_action)
            if game.check_win()[0] != 0: break
        
        if game.check_win()[0] == 0:
            game.switch_turn()
            # MCTS needs reset or tree trimming at turn change
            mcts_p1.reset()
            mcts_p2.reset()

    winner, total_points = game.check_win()
    return winner, total_points, history


# ---------------------------------------------------------
# Match Functions with Proper Reward Attribution
# ---------------------------------------------------------

def play_self_play_match(game, mcts, model, device, is_eval=False, cube_epsilon=0.0):
    """
    Plays a full match (Self-Play) until Config.MATCH_TARGET is reached.
    
    Args:
        game: BackgammonGame instance
        mcts: MCTS instance
        model: Neural network model
        device: torch device
        is_eval: If True, no exploration
        cube_epsilon: Exploration rate for cube decisions (0.0 in eval)
    
    Returns:
        full_match_history: List of training tuples
        overall_winner: 1 or -1
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
            is_eval=is_eval,
            cube_epsilon=cube_epsilon
        )

        scores[winner] += points
        
        # Calculate base reward magnitude from game outcome
        reward_mag = float(points) / (Config.MATCH_TARGET * (Config.R_BACKGAMMON + 1))
        
        # Process history with proper reward attribution
        for h in game_history:
            player_val = 1 if h['is_p1'] else -1
            is_winner = (player_val == winner)
            
            # CRITICAL FIX: Use immediate reward if set, otherwise game outcome
            if h.get('immediate_reward') is not None:
                # This was a cube decision with immediate resolution (drop)
                reward = h['immediate_reward']
            else:
                # Regular move or cube decision that led to continued play
                reward = reward_mag if is_winner else -reward_mag
            
            # Convert to training tuple format
            full_match_history.append((
                h['board'], 
                h['ctx'], 
                h['action'], 
                reward, 
                h['is_cube'], 
                h['probs']
            ))

    overall_winner = 1 if scores[1] >= Config.MATCH_TARGET else -1
    return full_match_history, overall_winner


def play_vs_baseline_match(game, current_model, baseline_model, mcts_current, device, cube_epsilon=0.0):
    """
    Plays a full match: Current Model vs Baseline.
    Only collects training data for the Current Model.
    
    Args:
        game: BackgammonGame instance
        current_model: Model being trained
        baseline_model: Fixed baseline model
        mcts_current: MCTS for current model
        device: torch device
        cube_epsilon: Exploration rate for cube decisions
    
    Returns:
        full_match_history: List of training tuples (only for current model)
        current_won_match: Boolean indicating if current model won
    """
    scores = {1: 0, -1: 0}
    full_match_history = []
    crawford_occurred = False
    
    # Randomly assign seats for the match
    current_is_p1 = random.choice([True, False])
    
    # Setup models/MCTS based on seats
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

        winner, points, game_history = _play_single_game(
            game=game,
            model_p1=model_p1, mcts_p1=mcts_p1,
            model_p2=model_p2, mcts_p2=mcts_p2,
            scores=scores,
            crawford_active=is_crawford_game,
            device=device,
            collect_history_p1=current_is_p1,      # Only collect if current model is P1
            collect_history_p2=(not current_is_p1), # Only collect if current model is P2
            is_eval=False,  # Allow exploration even vs baseline
            cube_epsilon=cube_epsilon
        )

        scores[winner] += points

        # Calculate Reward
        reward_mag = float(points) / (Config.MATCH_TARGET * (Config.R_BACKGAMMON + 1))
        
        # Add to history (game_history only contains current_model's moves due to flags above)
        for h in game_history:
            # Did the current model win this game?
            am_i_p1 = current_is_p1
            i_won = (winner == 1 and am_i_p1) or (winner == -1 and not am_i_p1)
            
            # Use immediate reward if available, otherwise game outcome
            if h.get('immediate_reward') is not None:
                # Adjust sign based on whether current model was the actor
                reward = h['immediate_reward'] if i_won else -h['immediate_reward']
            else:
                reward = reward_mag if i_won else -reward_mag
            
            full_match_history.append((
                h['board'], 
                h['ctx'], 
                h['action'], 
                reward, 
                h['is_cube'], 
                h['probs']
            ))

    current_won_match = (scores[1] >= Config.MATCH_TARGET) if current_is_p1 else (scores[-1] >= Config.MATCH_TARGET)
    return full_match_history, current_won_match