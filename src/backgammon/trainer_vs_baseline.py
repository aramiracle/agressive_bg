"""Train new model vs older baseline version."""

import os
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
from src.backgammon.trainer import play_one_game, finalize_history, get_cube_decision, train_batch


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
    Returns (training_data, did_current_win).
    Only saves training data for current model's decisions.
    """
    game.reset()
    current_is_p1 = random.choice([True, False])
    
    # Reset MCTS trees
    mcts_current.reset()
    mcts_baseline = MCTS(baseline_model, device)
    
    history = []
    
    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0:
            break
        
        is_current_turn = (game.turn == 1) == current_is_p1
        active_model = current_model if is_current_turn else baseline_model
        active_mcts = mcts_current if is_current_turn else mcts_baseline
        
        # --- CUBING PHASE ---
        if game.can_double() and game.cube < Config.MATCH_TARGET:
            double_choice = get_cube_decision(active_model, game, device)
            
            if double_choice == 1:
                # Opponent decides whether to take
                game.switch_turn()
                opponent_model = baseline_model if is_current_turn else current_model
                take_choice = get_cube_decision(opponent_model, game, device)
                game.switch_turn()
                
                # Save current model's doubling decision
                if is_current_turn:
                    board_t, ctx_t = game.get_vector(0, 0, device='cpu', canonical=True)
                    history.append((board_t, ctx_t, 1, game.turn, True))
                
                if take_choice == 1:
                    game.apply_double()
                else:
                    # Opponent refused - current player wins cube value
                    final_winner, points = game.handle_cube_refusal()
                    current_won = (final_winner == 1) == current_is_p1
                    # Adjust winner perspective for history
                    hist_winner = game.turn if current_won else -game.turn
                    return finalize_history(history, hist_winner, points), current_won
        
        # --- MOVEMENT PHASE ---
        game.roll_dice()
        
        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                game.dice = []
                break
            
            root = active_mcts.search(game, 0, 0)
            
            if root.children:
                actions = list(root.children.keys())
                visits = torch.tensor([c.visits for c in root.children.values()], dtype=torch.float)
                if visits.sum() > 0:
                    chosen_action = actions[torch.multinomial(visits / visits.sum(), 1).item()]
                else:
                    chosen_action = actions[0]
            else:
                chosen_action = random.choice(legal)
            
            # Only save current model's moves for training
            if is_current_turn:
                board_t, ctx_t = game.get_vector(0, 0, device='cpu', canonical=True)
                canon_act = game.real_action_to_canonical(chosen_action)
                s_idx, e_idx = move_to_indices(canon_act[0], canon_act[1])
                history.append((board_t, ctx_t, (s_idx, e_idx), game.turn, False))
            
            game.step_atomic(chosen_action)
            
            # Advance tree for active player
            active_mcts.advance_to_child(chosen_action)
            
            if game.check_win()[0] != 0:
                break
        
        if game.check_win()[0] == 0:
            game.switch_turn()
            # Reset trees on turn switch
            mcts_current.reset()
            mcts_baseline.reset()
    
    # --- FINALIZATION ---
    winner, mult = game.check_win()
    total_points = mult * game.cube
    current_won = (winner == 1) == current_is_p1
    
    # History is from current model's perspective
    # If current won, their moves get positive reward
    hist_winner = 1 if current_is_p1 else -1
    if not current_won:
        hist_winner = -hist_winner
    
    # Remap history turns to consistent winner reference
    data = []
    for board, ctx, act, turn, is_cube in history:
        # turn was saved as game.turn when current model played
        # current model always played when is_current_turn was True
        # so turn corresponds to current_is_p1's player number
        if current_won:
            reward = float(total_points) + float(Config.MATCH_TARGET)
        else:
            reward = -float(total_points) - float(Config.MATCH_TARGET)
        data.append((board, ctx, act, reward, is_cube))
    
    return data, current_won


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
    
    # Load baseline with its own config
    baseline_path = os.path.join(os.path.dirname(checkpoint_dir), Config.BASELINE_DIR, Config.BASELINE_MODEL_NAME)
    baseline_config_path = os.path.join(os.path.dirname(checkpoint_dir), Config.BASELINE_DIR, 'config.py')
    
    baseline_model = None
    baseline_elo = Config.INITIAL_ELO
    use_baseline = False
    
    if os.path.exists(baseline_path) and os.path.exists(baseline_config_path):
        baseline_model, baseline_elo = load_model_with_config(baseline_config_path, baseline_path, device)
        print(f"✅ Baseline loaded: {baseline_path} (ELO: {baseline_elo:.0f})")
        use_baseline = True
    else:
        print(f"⚠️  No baseline found at {baseline_path}, using self-play only")
    
    # Use prioritized replay buffer
    use_prioritized = getattr(Config, 'USE_PRIORITIZED_REPLAY', True)
    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=use_prioritized)
    
    game = BackgammonGame()
    
    if getattr(Config, 'BASELINE_SWITCH_ON_SURPASS', True):
        phase = "vs_baseline" if use_baseline and current_elo < baseline_elo else "self_play"
    else:
        phase = "vs_baseline" if use_baseline else "self_play"
    
    print(f"\n🎮 Training: Current ELO={current_elo:.0f}, Best ELO={best_elo:.0f}")
    if use_baseline:
        print(f"   Baseline ELO={baseline_elo:.0f}")
        ratio_pct = getattr(Config, 'BASELINE_SELF_PLAY_RATIO', 0.5) * 100
        if phase == "vs_baseline":
            print(f"   Phase: vs_baseline ({ratio_pct:.0f}% self-play, {100-ratio_pct:.0f}% vs baseline)\n")
        else:
            print(f"   Phase: self_play (100% self-play)\n")
    else:
        print(f"   Phase: self_play only\n")
    
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")
    
    while train_step < Config.TRAIN_STEPS:
        # 1. DATA COLLECTION PHASE
        model.eval()
        mcts = MCTS(model, device)
        
        baseline_ratio = getattr(Config, 'BASELINE_SELF_PLAY_RATIO', 0.5)
        if phase == "vs_baseline":
            num_self = int(Config.GAMES_PER_ITERATION * baseline_ratio)
            num_baseline = Config.GAMES_PER_ITERATION - num_self
        else:
            num_self = Config.GAMES_PER_ITERATION
            num_baseline = 0
        
        print(f"\n[Step {train_step}] Collecting {num_self} self-play + {num_baseline} vs baseline games...")
        
        # Self-play games
        for _ in range(num_self):
            data, _ = play_one_game(game, mcts, model, device)
            replay_buffer.extend(data)
        
        # Vs baseline games
        if use_baseline and num_baseline > 0:
            for _ in range(num_baseline):
                data, won = play_vs_baseline(game, model, baseline_model, mcts, device)
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
                
                if phase == "vs_baseline":
                    num_eval_self = int(Config.ELO_EVAL_GAMES * baseline_ratio)
                    num_eval_baseline = Config.ELO_EVAL_GAMES - num_eval_self
                else:
                    num_eval_self = Config.ELO_EVAL_GAMES
                    num_eval_baseline = 0
                
                wins_total = 0
                games_total = 0
                
                with torch.no_grad():
                    if num_eval_self > 0:
                        print(f"  Evaluating vs Best Model ({num_eval_self} games)...")
                        wins_self, total_self = evaluate_vs_opponent(
                            game, model, best_model,
                            num_eval_self, device,
                            show_progress=True
                        )
                        wins_total += wins_self
                        games_total += total_self
                        win_rate_self = (wins_self/total_self)*100 if total_self > 0 else 0
                        print(f"  → vs Best: {wins_self}/{total_self} wins ({win_rate_self:.1f}%)")
                    
                    if num_eval_baseline > 0 and baseline_model is not None:
                        print(f"  Evaluating vs Baseline ({num_eval_baseline} games)...")
                        wins_baseline, total_baseline = evaluate_vs_opponent(
                            game, model, baseline_model,
                            num_eval_baseline, device,
                            show_progress=True
                        )
                        wins_total += wins_baseline
                        games_total += total_baseline
                        win_rate_baseline = (wins_baseline/total_baseline)*100 if total_baseline > 0 else 0
                        print(f"  → vs Baseline: {wins_baseline}/{total_baseline} wins ({win_rate_baseline:.1f}%)")
                
                old_elo = current_elo
                
                if phase == "vs_baseline":
                    weight_best = baseline_ratio
                    weight_baseline = 1 - baseline_ratio
                    opponent_elo = (best_elo * weight_best) + (baseline_elo * weight_baseline)
                else:
                    opponent_elo = best_elo
                
                new_elo = update_elo(current_elo, opponent_elo, wins_total, games_total)
                
                print(f"\n📊 Evaluation Summary:")
                print(f"   Overall: {wins_total}/{games_total} wins ({(wins_total/games_total)*100:.1f}%)")
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
                
                if getattr(Config, 'BASELINE_SWITCH_ON_SURPASS', True) and use_baseline and phase == "vs_baseline" and current_elo >= baseline_elo:
                    phase = "self_play"
                    print(f"🎯 Surpassed baseline! Switching to 100% self-play (ELO {current_elo:.0f} >= {baseline_elo:.0f})")
                
                model.train()
        
        avg_loss = total_iter_loss / Config.STEPS_PER_ITERATION
        print(f"Iteration Complete. Avg Loss: {avg_loss:.4f} | Samples in Buffer: {len(replay_buffer)}")
    
    pbar.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    train()