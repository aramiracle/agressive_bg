"""
Train a new model using self-play with full cubing support and ELO tracking.
Features: GradScaler, replay buffer, evaluation vs best model, stable optimization,
multi-core self-play collection, and CUBE EXPLORATION FIXES.

Key changes for cube fix:
- Dynamic epsilon scheduling for cube exploration
- Pass cube_epsilon to game functions
- Optional cube curriculum learning
- Cube action logging for diagnostics
"""

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

from src.backgammon.config import Config
from src.backgammon.engine import BackgammonGame
from src.backgammon.model import get_model
from src.backgammon.mcts import MCTS
from src.backgammon.utils.checkpoint import (
    setup_checkpoint_dir, save_checkpoint, load_checkpoint,
    get_model_state_dict, load_model_state_dict
)
from src.backgammon.utils.elo import evaluate_vs_opponent, update_elo
from src.backgammon.replay_buffer import get_replay_buffer
from src.backgammon.utils.game import play_self_play_match
from src.backgammon.utils.train import train_batch

torch.multiprocessing.set_sharing_strategy("file_system")

# ------------------------------------------------------------
# Cube Epsilon Scheduling
# ------------------------------------------------------------

def get_cube_epsilon(train_step):
    """
    Calculate cube exploration rate based on training step.
    Linear decay from CUBE_EPSILON_START to CUBE_EPSILON_END.
    
    Returns:
        epsilon: float between 0 and 1
    """
    if Config.CUBE_CURRICULUM_ENABLED:
        # Use curriculum stages
        for i in reversed(range(len(Config.CUBE_CURRICULUM_STAGES))):
            stage = Config.CUBE_CURRICULUM_STAGES[i]
            if train_step >= stage['steps']:
                return stage['epsilon']
        return Config.CUBE_CURRICULUM_STAGES[0]['epsilon']
    else:
        # Simple linear decay
        if train_step >= Config.CUBE_EPSILON_DECAY_STEPS:
            return Config.CUBE_EPSILON_END
        
        progress = train_step / Config.CUBE_EPSILON_DECAY_STEPS
        epsilon = Config.CUBE_EPSILON_START - (Config.CUBE_EPSILON_START - Config.CUBE_EPSILON_END) * progress
        return epsilon

def get_cube_loss_weight(train_step):
    """
    Get cube loss weight based on training step (for curriculum learning).
    """
    if Config.CUBE_CURRICULUM_ENABLED:
        for i in reversed(range(len(Config.CUBE_CURRICULUM_STAGES))):
            stage = Config.CUBE_CURRICULUM_STAGES[i]
            if train_step >= stage['steps']:
                return stage['cube_weight']
        return Config.CUBE_CURRICULUM_STAGES[0]['cube_weight']
    else:
        return Config.CUBE_LOSS_WEIGHT

# ------------------------------------------------------------
# Worker for multi-core collection
# ------------------------------------------------------------

def collection_worker(args):
    """
    Worker function for parallel self-play collection.
    Now includes cube_epsilon parameter.
    """
    model_state, matches_per_worker, device, cube_epsilon = args
    torch.set_num_threads(1)

    # Initialize once per worker
    game = BackgammonGame()
    model = get_model().to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    mcts = MCTS(model, cpuct=Config.C_PUCT, num_sims=Config.NUM_SIMULATIONS, device=device)

    collected = []
    for _ in range(matches_per_worker):
        game.reset()
        # CRITICAL: Pass cube_epsilon to enable exploration
        data, _ = play_self_play_match(
            game, mcts, model, device, 
            is_eval=False, 
            cube_epsilon=cube_epsilon
        )
        collected.extend(data)

    return collected

def parallel_collect_self_play(model, replay_buffer, total_matches, device="cpu", cube_epsilon=0.0):
    """
    Collect self-play data in parallel with cube exploration.
    """
    collection_device = getattr(Config, 'SELF_PLAY_DEVICE', device)
    
    num_workers = mp.cpu_count()
    matches_per_worker = max(1, total_matches // num_workers)
    actual_matches = matches_per_worker * num_workers
    
    model_state = model.state_dict()
    ctx = mp.get_context("spawn")
    
    # Pass cube_epsilon to workers
    args_list = [
        (model_state, matches_per_worker, collection_device, cube_epsilon) 
        for _ in range(num_workers)
    ]

    collected = []

    with ctx.Pool(processes=num_workers) as pool:
        pbar = tqdm(total=actual_matches, desc="Collecting (matches)", dynamic_ncols=True)
        for result in pool.imap_unordered(collect_worker_wrapper, args_list):
            collected.extend(result)
            pbar.update(len(result) // 20 if len(result) > 20 else 1) 
        pbar.close()

    replay_buffer.extend(collected)

def collect_worker_wrapper(args):
    return collection_worker(args)

# ------------------------------------------------------------
# Cube Diagnostics (Optional)
# ------------------------------------------------------------

class CubeDiagnostics:
    """Track and log cube decision statistics."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.double_offered = 0
        self.double_passed = 0
        self.take_accepted = 0
        self.drop_accepted = 0
    
    def log_double(self, action):
        if action == 1:
            self.double_offered += 1
        else:
            self.double_passed += 1
    
    def log_take(self, action):
        if action == 1:
            self.take_accepted += 1
        else:
            self.drop_accepted += 1
    
    def get_stats(self):
        total_double_ops = self.double_offered + self.double_passed
        total_take_ops = self.take_accepted + self.drop_accepted
        
        double_rate = self.double_offered / total_double_ops if total_double_ops > 0 else 0.0
        take_rate = self.take_accepted / total_take_ops if total_take_ops > 0 else 0.0
        
        return {
            'double_rate': double_rate,
            'take_rate': take_rate,
            'double_ops': total_double_ops,
            'take_ops': total_take_ops
        }

# ------------------------------------------------------------
# TRAIN LOOP
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
    best_elo = cp_best['elo'] if cp_best else current_elo

    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=True, device=device)
    game = BackgammonGame()

    # Initialize cube diagnostics
    cube_diag = CubeDiagnostics() if Config.CUBE_LOGGING_ENABLED else None

    print(f"\n🎮 Training Start: Current ELO={current_elo:.0f}, Best ELO={best_elo:.0f}")
    print(f"   Cube Exploration: {Config.CUBE_EPSILON_START:.2f} → {Config.CUBE_EPSILON_END:.2f} over {Config.CUBE_EPSILON_DECAY_STEPS} steps")
    print(f"   Cube Loss Weight: {Config.CUBE_LOSS_WEIGHT:.1f}x")
    
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")

    while train_step < Config.TRAIN_STEPS:
        # ======================================
        # COLLECTION PHASE with Cube Exploration
        # ======================================
        cube_epsilon = get_cube_epsilon(train_step)
        
        parallel_collect_self_play(
            model,
            replay_buffer,
            Config.MATCHES_PER_ITERATION,
            device=device,
            cube_epsilon=cube_epsilon
        )

        if len(replay_buffer) < Config.BATCH_SIZE:
            continue

        # ======================================
        # TRAINING PHASE
        # ======================================
        model.train()
        for _ in range(Config.TRAIN_UPDATES_PER_ITER):
            # Update cube loss weight if using curriculum
            if Config.CUBE_CURRICULUM_ENABLED:
                Config.CUBE_LOSS_WEIGHT = get_cube_loss_weight(train_step)
            
            loss, gnorm = train_batch(
                model,
                optimizer,
                replay_buffer,
                Config.BATCH_SIZE,
                device,
                scaler
            )

            train_step += 1
            buf_fill = (len(replay_buffer) / Config.BUFFER_SIZE) * 100

            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'elo': f'{current_elo:.0f}',
                'gnorm': f'{gnorm:.2f}',
                'buf': f'{buf_fill:.1f}%',
                'ε_cube': f'{cube_epsilon:.3f}'
            })

            # ======================================
            # EVALUATION PHASE
            # ======================================
            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                print(f"\n\n🎯 ELO Evaluation at Step {train_step}")
                print(f"   Cube Epsilon: {cube_epsilon:.3f}")
                
                model.eval()
                best_model.eval()

                # Evaluation with NO exploration (epsilon=0)
                args = (game, model, best_model, Config.ELO_EVAL_GAMES, 'cpu', None)
                wins, total = evaluate_vs_opponent(args)

                old_elo = current_elo
                current_elo = update_elo(current_elo, best_elo, wins, total)
                print(f"  📊 Summary: {wins}/{total} wins | ELO: {old_elo:.0f} -> {current_elo:.0f}")

                if current_elo > best_elo:
                    print(f"  🏆 NEW BEST MODEL (ELO: {current_elo:.0f})")
                    best_elo = current_elo
                    load_model_state_dict(best_model, get_model_state_dict(model))
                    save_checkpoint(model, optimizer, train_step, best_elo, loss, best_path)

                save_checkpoint(model, optimizer, train_step, current_elo, loss, latest_path)
                
                # Log cube diagnostics if enabled
                if cube_diag is not None:
                    stats = cube_diag.get_stats()
                    print(f"  🎲 Cube Stats: Double Rate={stats['double_rate']:.1%}, Take Rate={stats['take_rate']:.1%}")
                    print(f"     Operations: {stats['double_ops']} doubles, {stats['take_ops']} takes")
                    cube_diag.reset()
                
                model.train()
                print("-" * 50)

    pbar.close()
    print("\n✅ Training complete!")
    print(f"Final Cube Epsilon: {get_cube_epsilon(train_step):.3f}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()