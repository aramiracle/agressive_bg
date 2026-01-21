# Trainer for Backgammon RL with Self-Taught Cubing and ELO tracking
# FULL STABLE VERSION (Crawford-safe, MCTS-safe, Cube-head fixed)

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
    """Returns 1 = double, 0 = no double"""
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)
    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))

        # Win probability from value head
        win_prob = torch.sigmoid(v.squeeze(0)).item()
        cube_choice = torch.argmax(cube_logits, dim=1).item()

    # Safety: only allow double if model thinks it's likely winning
    if win_prob < 0.55:
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
    """Plays a full backgammon game with cubing + MCTS

    Returns:
      - Training mode: (data, winner)
      - Eval mode: (winner, points)
    """

    # ---------------- Match state ----------------
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

    # ---------------- Main loop ----------------
    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0:
            break

        my_s = local_match_scores[game.turn]
        opp_s = local_match_scores[-game.turn]

        # ---------------- Cubing phase ----------------
        if game.can_double() and game.cube < Config.MATCH_TARGET:
            double_choice = get_cube_decision(model, game, device, my_s, opp_s)

            if double_choice == 1:
                # Opponent decision
                game.switch_turn()
                take_choice = get_cube_decision(model, game, device, opp_s, my_s)
                game.switch_turn()

                if not is_eval:
                    board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                    history.append((board_t, ctx_t, 1, game.turn, True))

                if take_choice == 1:
                    game.apply_double()
                else:
                    final_winner, points = game.handle_cube_refusal()

                    if is_eval:
                        if game.crawford_active:
                            game.crawford_used = True
                        return final_winner, points

                    return finalize_history(history, final_winner, points), final_winner

        # ---------------- Movement phase ----------------
        game.roll_dice()

        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                game.dice = []
                break

            root = mcts.search(game, my_s, opp_s)

            # Select action
            if root.children:
                actions = list(root.children.keys())
                visits = torch.tensor([c.visits for c in root.children.values()], dtype=torch.float)

                if visits.sum() > 0:
                    probs = visits / visits.sum()
                    idx = torch.multinomial(probs, 1).item()
                    chosen_action = actions[idx]
                else:
                    chosen_action = actions[0]
            else:
                chosen_action = random.choice(legal)

            # Save training data
            if not is_eval:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                canon_act = game.real_action_to_canonical(chosen_action)
                s_idx, e_idx = move_to_indices(canon_act[0], canon_act[1])
                history.append((board_t, ctx_t, (s_idx, e_idx), game.turn, False))

            # Apply move
            game.step_atomic(chosen_action)

            # Safe MCTS advance
            if not mcts.root or chosen_action not in mcts.root.children:
                mcts.reset()
            else:
                mcts.advance_to_child(chosen_action)

            if game.check_win()[0] != 0:
                break

        if game.check_win()[0] == 0:
            game.switch_turn()
            mcts.reset()

    # ---------------- Game end ----------------
    winner, mult = game.check_win()
    total_points = mult * game.cube

    if game.crawford_active:
        game.crawford_used = True
        game.crawford_active = False

    if is_eval:
        return winner, total_points

    return finalize_history(history, winner, total_points), winner


# ------------------------------------------------------------
# History → training samples
# ------------------------------------------------------------

def finalize_history(history, winner, total_points):
    data = []

    for board, ctx, act, turn, is_cube in history:
        if turn == winner:
            reward = float(total_points) + float(Config.MATCH_TARGET)
        else:
            reward = -float(total_points) - float(Config.MATCH_TARGET)

        reward /= float(Config.MATCH_TARGET)
        data.append((board, ctx, act, reward, is_cube))

    return data


# ------------------------------------------------------------
# Training step
# ------------------------------------------------------------

def train_batch(model, optimizer, replay_buffer, batch_size, device, scaler):
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0

    batch, indices, weights = replay_buffer.sample(batch_size)
    if not batch:
        return 0.0, 0.0

    boards = torch.stack([x[0] for x in batch]).to(device)
    contexts = torch.stack([x[1] for x in batch]).to(device)
    rewards = torch.tensor([x[3] for x in batch], dtype=torch.float, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float, device=device)

    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        p_from, p_to, v, cube_logits = model(boards, contexts)

        normalized_rewards = rewards
        td_errors = (v.squeeze(-1) - normalized_rewards).detach()

        v_loss = (weights_t * (v.squeeze(-1) - normalized_rewards) ** 2).mean()

        p_loss = torch.tensor(0.0, device=device)
        c_loss = torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0

        for i, (_, _, action, _, is_cube) in enumerate(batch):
            w = weights_t[i]

            if is_cube:
                with torch.no_grad():
                    win_prob = torch.sigmoid(v[i]).item()
                target = torch.tensor([1 if win_prob > 0.6 else 0], device=device)

                c_loss += w * nn.functional.cross_entropy(cube_logits[i:i+1], target)
                c_count += 1
            else:
                t_f = torch.tensor([action[0]], device=device)
                t_t = torch.tensor([action[1]], device=device)

                p_loss += w * (
                    nn.functional.cross_entropy(p_from[i:i+1], t_f) +
                    nn.functional.cross_entropy(p_to[i:i+1], t_t)
                ) * 0.5
                p_count += 1

        loss = v_loss
        if p_count > 0:
            loss += p_loss / p_count
        if c_count > 0:
            loss += c_loss / c_count

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)

    scaler.step(optimizer)
    scaler.update()

    replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

    return loss.item(), grad_norm.item()


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

def train():
    checkpoint_dir, best_path, latest_path = setup_checkpoint_dir()
    device = torch.device(Config.DEVICE)

    model = get_model().to(device)
    best_model = get_model().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    cp_latest = load_checkpoint(latest_path, model, optimizer, device)
    train_step = cp_latest['step'] if cp_latest else 0
    current_elo = cp_latest['elo'] if cp_latest else Config.INITIAL_ELO

    cp_best = load_checkpoint(best_path, best_model, None, device)
    if cp_best:
        best_elo = cp_best['elo']
    else:
        best_elo = current_elo
        load_model_state_dict(best_model, get_model_state_dict(model))

    use_prioritized = getattr(Config, 'USE_PRIORITIZED_REPLAY', True)
    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=use_prioritized)

    game = BackgammonGame()

    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")

    while train_step < Config.TRAIN_STEPS:
        model.eval()
        mcts = MCTS(model, device=device)

        # -------- Data collection --------
        for _ in range(Config.GAMES_PER_ITERATION):
            data, _ = play_one_game(game, mcts, model, device)
            replay_buffer.extend(data)

        if len(replay_buffer) < Config.BATCH_SIZE:
            continue

        # -------- Optimization --------
        model.train()
        total_iter_loss = 0.0

        for _ in range(Config.STEPS_PER_ITERATION):
            loss, gnorm = train_batch(
                model,
                optimizer,
                replay_buffer,
                Config.BATCH_SIZE,
                device,
                scaler
            )

            total_iter_loss += loss
            train_step += 1
            pbar.update(1)

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'gnorm': f'{gnorm:.2f}',
                'elo': f'{current_elo:.0f}',
                'buffer': f'{len(replay_buffer)}/{Config.BUFFER_SIZE}'
            })

            # -------- Evaluation --------
            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                model.eval()
                best_model.eval()

                with torch.no_grad():
                    wins, total = evaluate_vs_opponent(
                        game,
                        model,
                        best_model,
                        Config.ELO_EVAL_GAMES,
                        device,
                        True
                    )

                new_elo = update_elo(current_elo, best_elo, wins, total)

                print(f"\n📊 ELO: {current_elo:.0f} → {new_elo:.0f}")

                if new_elo > best_elo:
                    best_elo = new_elo
                    load_model_state_dict(best_model, get_model_state_dict(model))
                    save_checkpoint(model, optimizer, train_step, best_elo, loss, best_path)

                current_elo = new_elo
                save_checkpoint(model, optimizer, train_step, current_elo, loss, latest_path)

                model.train()

    pbar.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    train()