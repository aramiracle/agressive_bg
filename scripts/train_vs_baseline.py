#!/usr/bin/env python3
"""Train new model vs older baseline version."""

import sys
import os
import torch
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from backgammon.config import Config
from backgammon.engine import BackgammonGame
from backgammon.model import get_model
from backgammon.mcts import MCTS
from backgammon.checkpoint import setup_checkpoint_dir, save_checkpoint, load_checkpoint, get_model_state_dict, load_model_state_dict
from backgammon.elo import evaluate_vs_opponent, update_elo
from backgammon.trainer import play_one_game, train_batch


def load_model_with_config(config_path, model_path, device):
    """Load model with its own config without affecting global Config."""
    import importlib.util
    from backgammon.model import BackgammonTransformer, BackgammonCNN
    
    # Load config module
    spec = importlib.util.spec_from_file_location("model_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    ModelConfig = config_module.Config
    
    # Create model with its own config
    if ModelConfig.MODEL_TYPE == "transformer":
        model = BackgammonTransformer(config=ModelConfig).to(device)
    elif ModelConfig.MODEL_TYPE == "cnn":
        model = BackgammonCNN(config=ModelConfig).to(device)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {ModelConfig.MODEL_TYPE}")
    
    # Load checkpoint
    cp = load_checkpoint(model_path, model, None, device)
    elo = cp['elo'] if cp else ModelConfig.INITIAL_ELO
    model.eval()
    
    return model, elo


def play_vs_baseline(game, current_model, baseline_model, device):
    """Play game: current vs baseline. Returns (training_data, did_current_win)."""
    game.reset()
    current_is_p1 = random.choice([True, False])
    
    mcts_current = MCTS(current_model, device)
    
    # Baseline model always in eval mode with no_grad
    baseline_model.eval()
    with torch.no_grad():
        mcts_baseline = MCTS(baseline_model, device)
        history = []
        
        for _ in range(Config.MAX_GAME_MOVES):
            winner, _ = game.check_win()
            if winner != 0:
                break
            
            is_current_turn = (game.turn == 1) == current_is_p1
            mcts = mcts_current if is_current_turn else mcts_baseline
            
            # Simplified: no cubing in vs baseline games
            game.roll_dice()
            
            while game.dice:
                legal = game.get_legal_moves()
                if not legal:
                    game.dice = []
                    break
                
                root = mcts.search(game, 0, 0)
                
                if root.children:
                    actions = list(root.children.keys())
                    visits = torch.tensor([c.visits for c in root.children.values()], dtype=torch.float)
                    if visits.sum() > 0:
                        chosen_action = actions[torch.multinomial(visits / visits.sum(), 1).item()]
                    else:
                        chosen_action = actions[0]
                else:
                    chosen_action = legal[0]
                
                # Only save current model's moves
                if is_current_turn:
                    board_t, ctx_t = game.get_vector(0, 0, device='cpu', canonical=True)
                    from backgammon.utils import move_to_indices
                    canon_act = game.real_action_to_canonical(chosen_action)
                    s_idx, e_idx = move_to_indices(canon_act[0], canon_act[1])
                    history.append((board_t, ctx_t, (s_idx, e_idx), game.turn, False))
                
                game.step_atomic(chosen_action)
                if game.check_win()[0] != 0:
                    break
            
            if game.check_win()[0] == 0:
                game.switch_turn()
        
        winner, mult = game.check_win()
        total_points = mult * game.cube
        current_won = (winner == 1) == current_is_p1
    
    # Create training data
    data = []
    for board, ctx, act, turn, is_cube in history:
        reward = (float(total_points) + float(Config.MATCH_TARGET)) if turn == winner else (-float(total_points) - Config.MATCH_TARGET)
        data.append((board, ctx, act, reward, is_cube))
    
    return data, current_won


def train():
    checkpoint_dir, best_path, latest_path = setup_checkpoint_dir()
    device = torch.device(Config.DEVICE)
    
    # Current model (new config)
    model = get_model().to(device)
    best_model = get_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # Load current model
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
    
    if os.path.exists(baseline_path) and os.path.exists(baseline_config_path):
        baseline_model, baseline_elo = load_model_with_config(baseline_config_path, baseline_path, device)
        print(f"✅ Baseline loaded: {baseline_path} (ELO: {baseline_elo:.0f})")
        use_baseline = True
    else:
        print(f"⚠️  No baseline found at {baseline_path}, using self-play only")
        use_baseline = False
        baseline_elo = Config.INITIAL_ELO
    
    replay_buffer = deque(maxlen=Config.BUFFER_SIZE)
    game = BackgammonGame()
    
    # Adaptive training: use config ratio until surpassing, then switch to 100% self-play
    baseline_wins = 0
    baseline_games = 0
    
    if Config.BASELINE_SWITCH_ON_SURPASS:
        phase = "vs_baseline" if use_baseline and current_elo < baseline_elo else "self_play"
    else:
        phase = "vs_baseline" if use_baseline else "self_play"
    
    print(f"\n🎮 Training: Current ELO={current_elo:.0f}, Best ELO={best_elo:.0f}")
    if use_baseline:
        print(f"   Baseline ELO={baseline_elo:.0f}")
        ratio_pct = Config.BASELINE_SELF_PLAY_RATIO * 100
        if phase == "vs_baseline":
            print(f"   Phase: vs_baseline ({ratio_pct:.0f}% self-play, {100-ratio_pct:.0f}% vs baseline)\n")
        else:
            print(f"   Phase: self_play (100% self-play)\n")
    else:
        print(f"   Phase: self_play only\n")
    
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Training")
    
    while train_step < Config.TRAIN_STEPS:
        # Data collection
        model.eval()
        mcts = MCTS(model, device)
        
        # Adaptive ratio based on phase
        if phase == "vs_baseline":
            num_self = int(Config.GAMES_PER_ITERATION * Config.BASELINE_SELF_PLAY_RATIO)
            num_baseline = Config.GAMES_PER_ITERATION - num_self
        else:  # self_play phase
            num_self = Config.GAMES_PER_ITERATION
            num_baseline = 0
        
        # Self-play games
        for _ in range(num_self):
            data, _ = play_one_game(game, mcts, model, device)
            replay_buffer.extend(data)
        
        # Vs baseline games
        if use_baseline and num_baseline > 0:
            for _ in range(num_baseline):
                data, won = play_vs_baseline(game, model, baseline_model, device)
                replay_buffer.extend(data)
                baseline_games += 1
                if won:
                    baseline_wins += 1
        
        if len(replay_buffer) < Config.BATCH_SIZE:
            continue
        
        # Training
        model.train()
        total_loss = 0
        
        for _ in range(Config.STEPS_PER_ITERATION):
            loss, gnorm = train_batch(model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler)
            total_loss += loss
            train_step += 1
            pbar.update(1)
            
            postfix = {
                'loss': f'{loss:.4f}',
                'elo': f'{current_elo:.0f}',
                'phase': phase[:4],  # Show "vs_b" or "self"
                'buf': f'{len(replay_buffer)/Config.BUFFER_SIZE*100:.0f}%'
            }
            
            if use_baseline and baseline_games > 0:
                postfix['vs_base'] = f'{baseline_wins/baseline_games*100:.0f}%'
            
            pbar.set_postfix(postfix)
            
            # Evaluate vs best
            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                model.eval()
                best_model.eval()
                
                with torch.no_grad():
                    wins, total = evaluate_vs_opponent(game, model, best_model, Config.ELO_EVAL_GAMES, device, show_progress=True)
                
                new_elo = update_elo(current_elo, best_elo, wins, total)
                
                print(f"\nStep {train_step}: ELO {current_elo:.0f}->{new_elo:.0f} | WR: {wins/total*100:.0f}%", end='')
                
                if new_elo > best_elo:
                    print(" 🏆 NEW BEST")
                    best_elo = new_elo
                    load_model_state_dict(best_model, get_model_state_dict(model))
                    save_checkpoint(model, optimizer, train_step, best_elo, loss, best_path)
                else:
                    print()
                
                current_elo = new_elo
                save_checkpoint(model, optimizer, train_step, current_elo, loss, latest_path)
                
                # Phase transition: switch to self-play when surpassing baseline (if enabled)
                if Config.BASELINE_SWITCH_ON_SURPASS and use_baseline and phase == "vs_baseline" and current_elo >= baseline_elo:
                    phase = "self_play"
                    print(f"🎯 Surpassed baseline! Switching to 100% self-play (ELO {current_elo:.0f} >= {baseline_elo:.0f})")
                
                # Reset baseline stats
                if use_baseline:
                    baseline_wins = 0
                    baseline_games = 0
                
                model.train()
    
    pbar.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    train()
