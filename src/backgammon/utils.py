"""Utility functions for backgammon AI."""

import random
import torch
import torch.nn as nn
from src.backgammon.mcts import MCTS
from src.backgammon.config import Config
from checkpoint import load_checkpoint


def move_to_indices(start, end):
    """
    Convert a move (start, end) to policy head indices.
    
    Args:
        start: Starting position ('bar' or 0-23)
        end: Ending position ('off' or 0-23)
    
    Returns:
        Tuple of (start_idx, end_idx) for policy heads
    """
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
    """
    Convert policy head indices back to a move tuple.
    
    Args:
        start_idx: Start index from policy head
        end_idx: End index from policy head
    
    Returns:
        Tuple of (start, end) where start/end can be 'bar'/'off' or int
    """
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
    
    # Top half (points 13-24)
    top = " ".join(f"{board[i]:+3d}" for i in range(12, 24))
    lines.append(f"13-24: {top}")
    
    # Bottom half (points 1-12, reversed for display)
    bot = " ".join(f"{board[i]:+3d}" for i in range(11, -1, -1))
    lines.append(f" 1-12: {bot}")
    
    # Bar and off
    lines.append(f"Bar: W={bar[0]} B={bar[1]} | Off: W={off[0]} B={off[1]}")
    
    return "\n".join(lines)

def smooth_distribution(target, epsilon, num_classes):
    """
    Applies label smoothing to a target distribution.
    target: tensor of shape (N,)
    epsilon: smoothing factor (e.g., 0.1)
    """
    uniform = torch.ones_like(target) / num_classes
    return (1.0 - epsilon) * target + epsilon * uniform

def get_learned_cube_decision(model, game, device, my_score, opp_score, stochastic=True):
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        # We ignore 'v' here because we want the Policy Head (cube_logits) to decide
        _, _, _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        
        # cube_logits should have size 2: [Action_0, Action_1]
        # For the Doubler: [No Double, Double]
        # For the Taker:   [Drop, Take]
        probs = torch.softmax(cube_logits.squeeze(0), dim=0)

    if stochastic:
        # Sample during self-play for exploration
        action = torch.multinomial(probs, 1).item()
    else:
        # Pick best during evaluation
        action = torch.argmax(probs).item()

    return action, probs

def jensen_shannon_loss(log_p, q_target):
    """
    Computes Jensen-Shannon Divergence.
    log_p: Log-probabilities from the model (output of log_softmax)
    q_target: Probability distribution from MCTS (targets)
    """
    # Convert log_p back to probabilities for the mixture
    p = log_p.exp()
    
    # Calculate the midpoint distribution M
    m = 0.5 * (p + q_target)
    
    # KL(P || M)
    # Note: kl_div(input, target) in PyTorch expects input as log-probs
    loss_pm = nn.functional.kl_div(log_p, m, reduction='sum')
    
    # KL(Q || M)
    # We use q_target.log() here. Since q_target is smoothed, this is safe.
    loss_qm = nn.functional.kl_div(q_target.log(), m, reduction='sum')
    
    return 0.5 * (loss_pm + loss_qm)

def play_one_game(
    game,
    mcts,
    model,
    device,
    is_eval=False,
    initial_match_scores=None,
    initial_crawford_used=False
):
    if initial_match_scores is not None:
        match_scores = {1: int(initial_match_scores[0]), -1: int(initial_match_scores[1])}
    else:
        match_scores = {1: 0, -1: 0}

    game.reset()
    game.crawford_used = bool(initial_crawford_used)
    game.set_match_scores(match_scores[1], match_scores[-1])

    mcts.reset()
    history = []
    local_match_scores = {1: match_scores[1], -1: match_scores[-1]}

    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0:
            break

        my_s = local_match_scores[game.turn]
        opp_s = local_match_scores[-game.turn]
        cached_vector = game.get_vector(my_s, opp_s, device='cpu', canonical=True)

        # ---------------- Learned Cubing ----------------
        if game.can_double() and not game.crawford_active:
            # Current player's turn: Decide whether to offer a double
            double_choice, double_probs = get_learned_cube_decision(
                model, game, device, my_s, opp_s, stochastic=not is_eval
            )

            # Store the state and the probability distribution for training
            board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
            history.append((board_t, ctx_t, None, game.turn, True, double_probs))

            if double_choice == 1: # Model decided to double
                game.switch_turn()
                # Opponent's turn: Decide whether to Take or Drop
                take_choice, take_probs = get_learned_cube_decision(
                    model, game, device, opp_s, my_s, stochastic=not is_eval
                )
                
                # Store the opponent's cube decision too
                board_opp, ctx_opp = game.get_vector(opp_s, my_s, device='cpu', canonical=True)
                history.append((board_opp, ctx_opp, None, game.turn, True, take_probs))
                
                game.switch_turn() # Return turn to active player

                if take_choice == 1:
                    game.apply_double()
                else:
                    winner, points = game.handle_cube_refusal()
                    return finalize_history(history, winner, points), winner

        # ---------------- Movement ----------------
        game.roll_dice()

        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                game.dice = []
                break

            root = mcts.search(game, my_s, opp_s)
            children = root.children
            
            # 1. Calculate probabilities from visits
            visits = torch.tensor([c.visits for c in children], dtype=torch.float)
            if visits.sum() > 0:
                probs = visits / visits.sum()
                idx = torch.multinomial(probs, 1).item()
            else:
                probs = torch.ones(len(children)) / len(children)
                idx = 0

            chosen_action = children[idx].action

            if not is_eval:
                board_t, ctx_t = cached_vector
                
                # 2. PATCH: Create full-sized target distributions (Size 26)
                target_f = torch.zeros(26)
                target_t = torch.zeros(26)
                
                for i, child in enumerate(children):
                    c_act = game.real_action_to_canonical(child.action)
                    s_idx, e_idx = move_to_indices(c_act[0], c_act[1])
                    target_f[s_idx] += probs[i]
                    target_t[e_idx] += probs[i]

                # Store both target distributions
                history.append(
                    (board_t, ctx_t, None, game.turn, False, (target_f, target_t))
                )

            game.step_atomic(chosen_action)
            mcts.advance_to_child(chosen_action)

            if game.check_win()[0] != 0: break

        if game.check_win()[0] == 0:
            game.switch_turn()
            mcts.reset()

    winner, total_points = game.check_win()
    if is_eval: return winner, total_points * game.cube
    return finalize_history(history, winner, total_points * game.cube), winner

def finalize_history(history, current_won, total_points):
    """
    history: List of (board, ctx, action_taken, turn, is_cube, policy_probs)
    """
    data = []
    # Normalize points by match target
    reward_magnitude = (float(total_points) + Config.MATCH_TARGET) / Config.MATCH_TARGET
    final_reward = reward_magnitude if current_won else -reward_magnitude

    for board, ctx, act, turn, is_cube, probs in history:
        # Since history only contains Current Model decisions, 
        # the reward is always 'final_reward'
        data.append((board, ctx, act, final_reward, is_cube, probs))
    return data

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

        # Stability: Move to Float32
        p_from, p_to, v, cube_logits = p_from.float(), p_to.float(), v.float(), cube_logits.float()

        # 1. Value Loss
        td_errors = torch.abs(v.squeeze(-1) - rewards).detach()
        v_loss = (weights_t * (v.squeeze(-1) - rewards) ** 2).mean()

        p_loss, c_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0
        smoothing = Config.LABEL_SMOOTHING

        # 2. Distribution Losses (JSD)
        for i, (_, _, _, _, is_cube, visit_targets) in enumerate(batch):
            weight = weights_t[i]

            if is_cube:
                target = smooth_distribution(visit_targets.to(device).float(), smoothing, 2)
                logp = nn.functional.log_softmax(cube_logits[i], dim=0)
                
                # Use JSD instead of KL
                c_loss += weight * jensen_shannon_loss(logp, target)
                c_count += 1
            else:
                target_f, target_t = visit_targets
                tf = smooth_distribution(target_f.to(device).float(), smoothing, 26)
                tt = smooth_distribution(target_t.to(device).float(), smoothing, 26)

                logp_f = nn.functional.log_softmax(p_from[i], dim=0)
                logp_t = nn.functional.log_softmax(p_to[i], dim=0)

                # Use JSD for policy heads
                js_f = jensen_shannon_loss(logp_f, tf)
                js_t = jensen_shannon_loss(logp_t, tt)

                p_loss += weight * 0.5 * (js_f + js_t)
                p_count += 1

        # Average losses by counts to keep scales consistent
        loss = v_loss
        if p_count > 0: loss += (p_loss / p_count)
        if c_count > 0: loss += (c_loss / c_count)

    # 3. Optimization Step
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
    """Load model with its own config without affecting global Config."""
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

def play_vs_baseline(game, current_model, baseline_model, mcts_current, device):
    """
    Play game: current vs baseline with full cubing support.
    Only saves training data for current model's decisions.
    """
    game.reset()
    game.set_match_scores(0, 0)
    current_is_p1 = random.choice([True, False])

    mcts_current.reset()
    mcts_baseline = MCTS(baseline_model, device=device)
    history = []

    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0: break

        is_current_turn = (game.turn == 1) == current_is_p1
        active_model = current_model if is_current_turn else baseline_model
        active_mcts = mcts_current if is_current_turn else mcts_baseline
        
        my_score = game.match_scores[game.turn]
        opp_score = game.match_scores[-game.turn]

        # =========================
        # LEARNED CUBING PHASE
        # =========================
        if game.can_double() and not game.crawford_active:
            double_choice, d_probs = get_learned_cube_decision(
                active_model, game, device, my_score, opp_score, stochastic=True
            )

            if is_current_turn:
                board_t, ctx_t = game.get_vector(my_score, opp_score, device='cpu', canonical=True)
                history.append((board_t, ctx_t, double_choice, game.turn, True, d_probs))

            if double_choice == 1:
                game.switch_turn() 
                is_responder_current = (game.turn == 1) == current_is_p1
                resp_model = current_model if is_responder_current else baseline_model
                
                take_choice, t_probs = get_learned_cube_decision(
                    resp_model, game, device, opp_score, my_score, stochastic=True
                )

                if is_responder_current:
                    board_t, ctx_t = game.get_vector(opp_score, my_score, device='cpu', canonical=True)
                    history.append((board_t, ctx_t, take_choice, game.turn, True, t_probs))

                game.switch_turn() 

                if take_choice == 1:
                    game.apply_double()
                else:
                    winner, points = game.handle_cube_refusal()
                    current_won = (winner == 1) == current_is_p1
                    return finalize_history(history, current_won, points), current_won

        # =========================
        # MOVEMENT PHASE
        # =========================
        game.roll_dice()
        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                game.dice = []
                break

            root = active_mcts.search(game, 0, 0)
            if root.children:
                visits = torch.tensor([c.visits for c in root.children], dtype=torch.float)
                probs = visits / visits.sum() if visits.sum() > 0 else torch.ones(len(visits))/len(visits)
                chosen_action = root.children[torch.multinomial(probs, 1).item()].action
            else:
                chosen_action = random.choice(legal)

            if is_current_turn:
                board_t, ctx_t = game.get_vector(game.match_scores[1], game.match_scores[-1], device='cpu', canonical=True)
                target_f, target_t = torch.zeros(26), torch.zeros(26)

                if root.children:
                    for i, child in enumerate(root.children):
                        canon_act = game.real_action_to_canonical(child.action)
                        s, e = move_to_indices(canon_act[0], canon_act[1])
                        if 0 <= s < 26: target_f[s] += probs[i]
                        if 0 <= e < 26: target_t[e] += probs[i]
                
                history.append((board_t, ctx_t, None, game.turn, False, (target_f, target_t)))

            game.step_atomic(chosen_action)
            active_mcts.advance_to_child(chosen_action)
            if game.check_win()[0] != 0: break

        if game.check_win()[0] == 0:
            game.switch_turn()
            mcts_current.reset()
            mcts_baseline.reset()

    winner, mult = game.check_win()
    total_points = mult * game.cube
    current_won = (winner == 1) == current_is_p1

    return finalize_history(history, current_won, total_points), current_won