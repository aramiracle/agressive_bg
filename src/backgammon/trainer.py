# Full Trainer for Backgammon RL (Self-Play Only, ELO tracking, Multi-Core)
"""
Train a new model using self-play with full cubing support and ELO tracking.
Features: GradScaler, replay buffer, evaluation vs best model, stable optimization,
and multi-core self-play collection using torch.multiprocessing.
"""

import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

from src.backgammon.config import Config
from src.backgammon.engine import BackgammonGame
from src.backgammon.model import get_model
from src.backgammon.mcts import MCTS
from backgammon.utils.checkpoint import (
    setup_checkpoint_dir, save_checkpoint, load_checkpoint,
    get_model_state_dict, load_model_state_dict
)
from src.backgammon.utils.elo import evaluate_vs_opponent, update_elo
from src.backgammon.replay_buffer import get_replay_buffer
# UPDATED IMPORT: Using match-based function
from src.backgammon.utils.game import play_self_play_match
from src.backgammon.utils.train import train_batch

torch.multiprocessing.set_sharing_strategy("file_system")

# ------------------------------------------------------------
# Worker for multi-core collection
# ------------------------------------------------------------

def collection_worker(args):
    """
    Multiprocessing worker for self-play matches.
    """
    model_state, matches_per_worker, device = args

    torch.set_num_threads(1)

    game = BackgammonGame()
    model = get_model().to(device)
    model.load_state_dict(model_state)
    model.eval()
    mcts = MCTS(model, device=device)

    collected = []
    # Loop now represents "Matches", not single games
    for _ in range(matches_per_worker):
        # UPDATED: Call play_self_play_match instead of play_one_game
        data, _ = play_self_play_match(game, mcts, model, device)
        collected.extend(data)

    return collected

def collect_worker_wrapper(args):
    return collection_worker(args)

# ------------------------------------------------------------
# Parallel self-play collection
# ------------------------------------------------------------

def parallel_collect_self_play(model, replay_buffer, total_matches, device="cpu"):
    """
    Parallel self-play collection using imap_unordered with tqdm.
    """
    num_workers = mp.cpu_count()
    # Ensure at least 1 match per worker if total_matches < num_workers
    matches_per_worker = max(1, total_matches // num_workers)
    
    model_state = model.state_dict()
    ctx = mp.get_context("spawn")
    
    args_list = [(model_state, matches_per_worker, device) for _ in range(num_workers)]

    collected = []

    with ctx.Pool(processes=num_workers) as pool:
        pbar = tqdm(total=total_matches, desc="Collecting (matches)", dynamic_ncols=True)
        for result in pool.imap_unordered(collect_worker_wrapper, args_list):
            collected.extend(result)
            pbar.update(len(result) // 20 if len(result) > 20 else 1) # Approximate progress or specific count if possible
        pbar.close()

    replay_buffer.extend(collected)

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

    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=True)
    game = BackgammonGame()

    print(f"\n🎮 Training Start: Current ELO={current_elo:.0f}, Best ELO={best_elo:.0f}")
    phase = "self_play"
    print(f"   Phase={phase}")

    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")

    while train_step < Config.TRAIN_STEPS:
        # -------- COLLECTION PHASE (DECOUPLED) --------
        if train_step % Config.COLLECTION_INTERVAL == 0:
            parallel_collect_self_play(
                model,
                replay_buffer,
                Config.MATCHES_PER_ITERATION # Acts as 'Matches per iteration' now
            )

        if len(replay_buffer) < Config.BATCH_SIZE:
            continue

        # -------- TRAINING PHASE (HEAVY REUSE) --------
        model.train()
        for _ in range(Config.TRAIN_UPDATES_PER_ITER):
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
                'buf': f'{buf_fill:.1f}%'
            })

            # -------- EVALUATION PHASE --------
            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                print(f"\n\n🎯 ELO Evaluation at Step {train_step} (Phase: {phase})")
                model.eval()
                best_model.eval()

                # Note: evaluate_vs_opponent in elo.py should ideally handle match play 
                # or single games. Assuming it handles the logic correctly.
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
                model.train()
                print("-" * 50)

    pbar.close()
    print("\n✅ Training complete!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()