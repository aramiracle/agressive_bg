# Trainer for Backgammon RL with Self-Taught Cubing and ELO tracking
# FULL STABLE VERSION (Patched for Tensor Mismatches)

import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

from src.backgammon.config import Config
from src.backgammon.engine import BackgammonGame
from src.backgammon.model import get_model
from src.backgammon.mcts import MCTS
from src.backgammon.checkpoint import (
    setup_checkpoint_dir, save_checkpoint, load_checkpoint,
    get_model_state_dict, load_model_state_dict
)
from src.backgammon.elo import evaluate_vs_opponent, update_elo
from src.backgammon.utils import move_to_indices
from src.backgammon.replay_buffer import get_replay_buffer

# ------------------------------------------------------------
# Cube decision helper
# ------------------------------------------------------------

def get_cube_decision(model, game, device, my_score=0, opp_score=0):
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)
    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        win_prob = torch.sigmoid(v.squeeze(0)).item()
        cube_choice = torch.argmax(cube_logits, dim=1).item()

    if win_prob < 0.6:
        return 0
    return cube_choice

# ------------------------------------------------------------
# Self-play / Evaluation game
# ------------------------------------------------------------

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

        # ---------------- Cubing ----------------
        if game.can_double() and game.cube < Config.MATCH_TARGET:
            double_choice = get_cube_decision(model, game, device, my_s, opp_s)

            if double_choice == 1:
                game.switch_turn()
                take_choice = get_cube_decision(model, game, device, opp_s, my_s)
                game.switch_turn()

                if not is_eval:
                    board_t, ctx_t = cached_vector
                    # Store as 1.0 (double) or 0.0 (no double) for target
                    history.append((board_t, ctx_t, double_choice, game.turn, True, None))

                if take_choice == 1:
                    game.apply_double()
                else:
                    final_winner, points = game.handle_cube_refusal()
                    if is_eval: return final_winner, points
                    return finalize_history(history, final_winner, points), final_winner

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


# ------------------------------------------------------------
# History → training samples
# ------------------------------------------------------------

def finalize_history(history, winner, total_points):
    data = []
    for board, ctx, act, turn, is_cube, visit_probs in history:
        # Normalize reward relative to match target
        reward = (float(total_points) + Config.MATCH_TARGET) if turn == winner else (-float(total_points) - Config.MATCH_TARGET)
        reward /= float(Config.MATCH_TARGET)
        data.append((board, ctx, act, reward, is_cube, visit_probs))
    return data


# ------------------------------------------------------------
# Training step
# ------------------------------------------------------------

def train_batch(model, optimizer, replay_buffer, batch_size, device, scaler):
    if len(replay_buffer) < batch_size: return 0.0, 0.0

    batch, indices, weights = replay_buffer.sample(batch_size)
    
    boards = torch.stack([x[0] for x in batch]).to(device)
    contexts = torch.stack([x[1] for x in batch]).to(device)
    rewards = torch.tensor([x[3] for x in batch], dtype=torch.float, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float, device=device)

    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        p_from, p_to, v, cube_logits = model(boards, contexts)

        # Value Loss
        td_errors = torch.abs(v.squeeze(-1) - rewards).detach()
        v_loss = (weights_t * (v.squeeze(-1) - rewards) ** 2).mean()

        p_loss = torch.tensor(0.0, device=device)
        c_loss = torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0

        for i, (_, _, action, reward, is_cube, visit_targets) in enumerate(batch):
            w = weights_t[i]

            if is_cube:
                # Cube target: 1 if this move led to a win, 0 otherwise
                target = torch.tensor([1 if reward > 0 else 0], device=device)
                c_loss += w * nn.functional.cross_entropy(cube_logits[i:i+1], target)
                c_count += 1
            else:
                if visit_targets is None: continue
                
                # PATCH: Use the two 26-sized target vectors
                target_f, target_t = visit_targets
                
                logp_f = nn.functional.log_softmax(p_from[i], dim=0)
                logp_t = nn.functional.log_softmax(p_to[i], dim=0)

                kl_f = nn.functional.kl_div(logp_f, target_f.to(device), reduction='batchmean')
                kl_t = nn.functional.kl_div(logp_t, target_t.to(device), reduction='batchmean')

                p_loss += w * 0.5 * (kl_f + kl_t)
                p_count += 1

        loss = v_loss
        if p_count > 0: loss += (p_loss / p_count)
        if c_count > 0: loss += (c_loss / c_count)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()

    replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
    return loss.item(), grad_norm.item()

# ------------------------------------------------------------
# Training loop (Remains unchanged but included for completeness)
# ------------------------------------------------------------

def train():
    checkpoint_dir, best_path, latest_path = setup_checkpoint_dir()
    device = torch.device(Config.DEVICE)

    model = get_model().to(device)
    best_model = get_model().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    cp_latest = load_checkpoint(latest_path, model, optimizer, device)
    train_step = cp_latest['step'] if cp_latest else 0
    current_elo = cp_latest['elo'] if cp_latest else Config.INITIAL_ELO

    cp_best = load_checkpoint(best_path, best_model, None, device)
    best_elo = cp_best['elo'] if cp_best else current_elo

    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=True)
    game = BackgammonGame()

    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")

    while train_step < Config.TRAIN_STEPS:
        model.eval()
        mcts = MCTS(model, device=device)

        for _ in range(Config.GAMES_PER_ITERATION):
            data, _ = play_one_game(game, mcts, model, device)
            replay_buffer.extend(data)

        if len(replay_buffer) < Config.BATCH_SIZE:
            continue

        model.train()
        for _ in range(Config.STEPS_PER_ITERATION):
            loss, gnorm = train_batch(model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler)
            train_step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss:.4f}', 'elo': f'{current_elo:.0f}'})

            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                model.eval()
                wins, total = evaluate_vs_opponent(game, model, best_model, Config.ELO_EVAL_GAMES, device, True)
                new_elo = update_elo(current_elo, best_elo, wins, total)
                if new_elo > best_elo:
                    best_elo = new_elo
                    load_model_state_dict(best_model, get_model_state_dict(model))
                    save_checkpoint(model, optimizer, train_step, best_elo, loss, best_path)
                current_elo = new_elo
                save_checkpoint(model, optimizer, train_step, current_elo, loss, latest_path)
                model.train()

    pbar.close()

if __name__ == "__main__":
    train()