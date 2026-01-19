"""
Trainer for Backgammon RL with Self-Taught Cubing and ELO tracking.
"""

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


def get_cube_decision(model, game, device):
    """Queries the model's cube head. returns 1 for Yes (Double/Take), 0 for No."""
    board_t, ctx_t = game.get_vector(device=device, canonical=True)
    with torch.no_grad():
        _, _, _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
    return torch.argmax(cube_logits, dim=1).item()


def play_one_game(game, mcts, model, device, is_eval=False, initial_match_scores=None, initial_crawford_used=False):
    """
    Plays a complete game with MCTS tree reuse.
    """
    if initial_match_scores is not None:
        match_scores = {1: int(initial_match_scores[0]), -1: int(initial_match_scores[1])}
    else:
        match_scores = {1: 0, -1: 0}

    game.crawford_used = bool(initial_crawford_used)
    game.reset()
    game.set_match_scores(match_scores[1], match_scores[-1])

    # Reset MCTS tree for new game
    mcts.reset()

    history = []
    local_match_scores = {1: match_scores[1], -1: match_scores[-1]}
    max_moves = Config.MAX_GAME_MOVES
    
    for _ in range(max_moves):
        winner, _ = game.check_win()
        if winner != 0:
            break

        my_s = local_match_scores[game.turn]
        opp_s = local_match_scores[-game.turn]

        # --- CUBING PHASE ---
        if game.can_double() and game.cube < Config.MATCH_TARGET:
            double_choice = get_cube_decision(model, game, device)
            
            if double_choice == 1:
                game.switch_turn()
                take_choice = get_cube_decision(model, game, device)
                game.switch_turn() 

                if not is_eval:
                    board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                    history.append((board_t, ctx_t, 1, game.turn, True))

                if take_choice == 1:
                    game.apply_double()
                else:
                    final_winner, points = game.handle_cube_refusal()
                    if is_eval: 
                        if game.crawford:
                            game.crawford_used = True
                            game.crawford = False
                        return final_winner, points
                    return finalize_history(history, final_winner, points), final_winner

        # --- MOVEMENT PHASE ---
        game.roll_dice()
        
        while game.dice:
            legal = game.get_legal_moves()
            if not legal: 
                game.dice = []
                break
            
            root = mcts.search(game, my_s, opp_s)
            
            if root.children:
                actions = list(root.children.keys())
                visits = torch.tensor([child.visits for child in root.children.values()], dtype=torch.float)
                
                if visits.sum() > 0:
                    probs = visits / visits.sum()
                    idx = torch.multinomial(probs, 1).item()
                    chosen_action = actions[idx]
                else:
                    priors = torch.tensor([child.prior for child in root.children.values()], dtype=torch.float)
                    idx = torch.multinomial(priors, 1).item()
                    chosen_action = actions[idx]
            else:
                chosen_action = random.choice(legal)

            if not is_eval:
                board_t, ctx_t = game.get_vector(my_s, opp_s, device='cpu', canonical=True)
                canon_act = game.real_action_to_canonical(chosen_action)
                s_idx, e_idx = move_to_indices(canon_act[0], canon_act[1])
                history.append((board_t, ctx_t, (s_idx, e_idx), game.turn, False))
            
            game.step_atomic(chosen_action)
            
            # Advance MCTS tree
            mcts.advance_to_child(chosen_action)
            
            if game.check_win()[0] != 0:
                break
        
        if game.check_win()[0] == 0:
            game.switch_turn()
            # Reset tree on turn switch (opponent's move invalidates our tree)
            mcts.reset()

    winner, mult = game.check_win()
    total_points = mult * game.cube

    if game.crawford:
        game.crawford_used = True
        game.crawford = False

    if is_eval: 
        return winner, total_points
        
    return finalize_history(history, winner, total_points), winner


def finalize_history(history, winner, total_points):
    data = []
    for board, ctx, act, turn, is_cube in history:
        reward = (float(total_points) + float(Config.MATCH_TARGET)) if turn == winner else (-float(total_points) - Config.MATCH_TARGET)
        data.append((board, ctx, act, reward, is_cube))
    return data


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
        
        normalized_rewards = rewards / Config.MATCH_TARGET 
        
        # Per-sample TD errors for priority update
        td_errors = (v.squeeze(-1) - normalized_rewards).detach()
        
        # Weighted MSE loss
        v_loss = (weights_t * (v.squeeze(-1) - normalized_rewards) ** 2).mean()
        
        p_loss, c_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0

        for i, (_, _, action, _, is_cube) in enumerate(batch):
            w = weights_t[i]
            if is_cube:
                target = torch.tensor([1 if rewards[i] > 0 else 0], device=device)
                c_loss += w * nn.functional.cross_entropy(cube_logits[i:i+1], target)
                c_count += 1
            else:
                t_f = torch.tensor([action[0]], device=device)
                t_t = torch.tensor([action[1]], device=device)
                p_loss += w * (nn.functional.cross_entropy(p_from[i:i+1], t_f) + 
                              nn.functional.cross_entropy(p_to[i:i+1], t_t)) * 0.5
                p_count += 1
        
        loss = v_loss + (p_loss / max(1, p_count)) + (c_loss / max(1, c_count))

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    
    # Update priorities
    replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
    
    return loss.item(), grad_norm.item()


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
    if cp_best: 
        best_elo = cp_best['elo']
    else:
        best_elo = current_elo
        load_model_state_dict(best_model, get_model_state_dict(model))

    # Use prioritized replay buffer
    use_prioritized = getattr(Config, 'USE_PRIORITIZED_REPLAY', True)
    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=use_prioritized)
    
    game = BackgammonGame()
    
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")
    
    while train_step < Config.TRAIN_STEPS:
        # 1. DATA COLLECTION PHASE
        model.eval()
        mcts = MCTS(model, device)
        
        print(f"\n[Step {train_step}] Collecting self-play games...")
        for _ in range(Config.GAMES_PER_ITERATION):
            data, _ = play_one_game(game, mcts, model, device)
            replay_buffer.extend(data)

        if len(replay_buffer) < Config.BATCH_SIZE:
            print(f"Buffer warming up: {len(replay_buffer)}/{Config.BATCH_SIZE}")
            continue

        # 2. OPTIMIZATION PHASE
        model.train()
        total_iter_loss = 0
        
        for i in range(Config.STEPS_PER_ITERATION):
            loss, gnorm = train_batch(model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler)
            total_iter_loss += loss
            train_step += 1
            pbar.update(1)

            buf_fill = (len(replay_buffer) / Config.BUFFER_SIZE) * 100
            pbar.set_postfix({
                'loss': f'{loss:.4f}', 
                'gnorm': f'{gnorm:.2f}',
                'elo': f'{current_elo:.0f}', 
                'buffer': f'{buf_fill:.1f}%'
            })

            # 3. EVALUATION PHASE
            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                print(f"\n{'='*60}")
                print(f"🎯 ELO Evaluation at Step {train_step}")
                print(f"{'='*60}")
                model.eval()
                best_model.eval()
                
                with torch.no_grad():
                    print(f"  Playing {Config.ELO_EVAL_GAMES} games vs Best Model...")
                    wins, total = evaluate_vs_opponent(
                        game, model, best_model,
                        Config.ELO_EVAL_GAMES, device,
                        show_progress=True
                    )

                old_elo = current_elo
                new_elo = update_elo(current_elo, best_elo, wins, total)
                
                print(f"\n📊 Evaluation Summary:")
                print(f"   Wins: {wins}/{total} ({(wins/total)*100:.1f}%)")
                print(f"   ELO: {old_elo:.0f} → {new_elo:.0f} (Δ{new_elo-old_elo:+.0f})")
                
                status_msg = "🏆 NEW BEST MODEL" if new_elo > best_elo else "No improvement"
                print(f"   {status_msg}")
                print(f"{'='*60}\n")

                if new_elo > best_elo:
                    best_elo = new_elo
                    load_model_state_dict(best_model, get_model_state_dict(model))
                    save_checkpoint(model, optimizer, train_step, best_elo, loss, best_path)
                
                current_elo = new_elo
                save_checkpoint(model, optimizer, train_step, current_elo, loss, latest_path)
                model.train()
        
        avg_loss = total_iter_loss / Config.STEPS_PER_ITERATION
        print(f"Iteration Complete. Avg Loss: {avg_loss:.4f} | Samples in Buffer: {len(replay_buffer)}")

    pbar.close()


if __name__ == "__main__":
    train()