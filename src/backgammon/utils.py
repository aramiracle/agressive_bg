"""Utility functions for backgammon AI."""

import random
import torch
import torch.nn as nn
from src.backgammon.mcts import MCTS
from src.backgammon.config import Config
from src.backgammon.checkpoint import load_checkpoint

# ---------------------------------------------------------
# Helper / Formatting Functions
# ---------------------------------------------------------

def move_to_indices(start, end):
    """Convert a move (start, end) to policy head indices."""
    if start == 'bar':
        start_idx = Config.BAR_IDX
    else:
        start_idx = start
    
    if end == 'off':
        end_idx = Config.OFF_IDX
    else:
        end_idx = end
    
    return start_idx, end_idx

def indices_to_move(start_idx, end_idx):
    """Convert policy head indices back to a move tuple."""
    if start_idx == Config.BAR_IDX:
        start = 'bar'
    else:
        start = start_idx
    
    if end_idx == Config.OFF_IDX:
        end = 'off'
    else:
        end = end_idx
    
    return start, end

def format_move(move):
    """Format a move for display."""
    start, end = move
    start_str = "BAR" if start == 'bar' else str(start + 1)
    end_str = "OFF" if end == 'off' else str(end + 1)
    return f"{start_str}->{end_str}"

def format_board(board, bar, off):
    """Format board state for display."""
    lines = []
    top = " ".join(f"{board[i]:+3d}" for i in range(12, 24))
    lines.append(f"13-24: {top}")
    bot = " ".join(f"{board[i]:+3d}" for i in range(11, -1, -1))
    lines.append(f" 1-12: {bot}")
    lines.append(f"Bar: W={bar[0]} B={bar[1]} | Off: W={off[0]} B={off[1]}")
    return "\n".join(lines)

def smooth_distribution(target, epsilon, num_classes):
    """Applies label smoothing to a target distribution."""
    uniform = torch.ones_like(target) / num_classes
    return (1.0 - epsilon) * target + epsilon * uniform

def get_learned_cube_decision(model, game, device, my_score, opp_score, stochastic=True):
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        _, _, _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        probs = torch.softmax(cube_logits.squeeze(0), dim=0)

    if stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    return action, probs

def jensen_shannon_loss(log_p, q_target):
    """Computes Jensen-Shannon Divergence."""
    p = log_p.exp()
    m = 0.5 * (p + q_target)
    loss_pm = nn.functional.kl_div(log_p, m, reduction='sum')
    loss_qm = nn.functional.kl_div(q_target.log(), m, reduction='sum')
    return 0.5 * (loss_pm + loss_qm)

def finalize_history(history, current_won, total_points):
    """
    Finalizes a game segment by assigning rewards.
    history: List of (board, ctx, action_taken, turn, is_cube, policy_probs)
    """
    data = []
    # Normalize points by match target
    reward_magnitude = (float(total_points) + Config.MATCH_TARGET) / Config.MATCH_TARGET
    final_reward = reward_magnitude if current_won else -reward_magnitude

    for board, ctx, act, turn, is_cube, probs in history:
        data.append((board, ctx, act, final_reward, is_cube, probs))
    return data

# ---------------------------------------------------------
# Core Game Logic (Single Game)
# ---------------------------------------------------------

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
    mcts_p1 = mcts_current if current_is_p1 else MCTS(baseline_model, device)
    
    model_p2 = baseline_model if current_is_p1 else current_model
    mcts_p2 = MCTS(baseline_model, device) if current_is_p1 else mcts_current

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


# ---------------------------------------------------------
# Training Functions (Unchanged)
# ---------------------------------------------------------

def train_batch(model, optimizer, replay_buffer, batch_size, device, scaler):
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0
    
    batch, indices, weights = replay_buffer.sample(batch_size)
    
    boards = torch.stack([x[0] for x in batch]).to(device)
    contexts = torch.stack([x[1] for x in batch]).to(device)
    rewards = torch.tensor([x[3] for x in batch], dtype=torch.float, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float, device=device)

    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        p_from, p_to, v, cube_logits = model(boards, contexts)
        p_from, p_to, v, cube_logits = p_from.float(), p_to.float(), v.float(), cube_logits.float()

        td_errors = torch.abs(v.squeeze(-1) - rewards).detach()
        v_loss = (weights_t * (v.squeeze(-1) - rewards) ** 2).mean()

        p_loss, c_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0
        smoothing = Config.LABEL_SMOOTHING

        for i, (_, _, _, _, is_cube, visit_targets) in enumerate(batch):
            weight = weights_t[i]

            if is_cube:
                target = smooth_distribution(visit_targets.to(device).float(), smoothing, 2)
                logp = nn.functional.log_softmax(cube_logits[i], dim=0)
                c_loss += weight * jensen_shannon_loss(logp, target)
                c_count += 1
            else:
                target_f, target_t = visit_targets
                tf = smooth_distribution(target_f.to(device).float(), smoothing, 26)
                tt = smooth_distribution(target_t.to(device).float(), smoothing, 26)

                logp_f = nn.functional.log_softmax(p_from[i], dim=0)
                logp_t = nn.functional.log_softmax(p_to[i], dim=0)

                js_f = jensen_shannon_loss(logp_f, tf)
                js_t = jensen_shannon_loss(logp_t, tt)
                p_loss += weight * 0.5 * (js_f + js_t)
                p_count += 1

        loss = v_loss
        if p_count > 0: loss += (p_loss / p_count)
        if c_count > 0: loss += (c_loss / c_count)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)

    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
        scaler.update()
        replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
    else:
        scaler.update()
        return loss.item(), 0.0

    return loss.item(), grad_norm.item()

def load_model_with_config(config_path, model_path, device):
    import importlib.util
    from src.backgammon.model import BackgammonTransformer, BackgammonCNN

    spec = importlib.util.spec_from_file_location("model_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    ModelConfig = config_module.Config

    if ModelConfig.MODEL_TYPE == "transformer":
        model = BackgammonTransformer(config=ModelConfig).to(device)
    elif ModelConfig.MODEL_TYPE == "cnn":
        model = BackgammonCNN(config=ModelConfig).to(device)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {ModelConfig.MODEL_TYPE}")

    cp = load_checkpoint(model_path, model, None, device)
    elo = cp['elo'] if cp else ModelConfig.INITIAL_ELO
    model.eval()

    return model, elo