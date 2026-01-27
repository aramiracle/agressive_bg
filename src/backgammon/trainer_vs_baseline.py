# Full Trainer vs Baseline (Crawford-safe, MCTS-safe, Engine-compatible)
"""
Train new model vs older baseline version with full cubing, MCTS reuse, and ELO tracking.
Features: Mixed-opponent collection, detailed evaluation vs both Best and Baseline models.
"""

import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp

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
# UPDATED IMPORTS: Using match-based functions
from src.backgammon.utils import play_self_play_match, play_vs_baseline_match, train_batch, load_model_with_config
from src.backgammon.replay_buffer import get_replay_buffer

torch.multiprocessing.set_sharing_strategy("file_system")

# ------------------------------------------------------------
# Collection helpers (Self-play + Vs Baseline)
# ------------------------------------------------------------

def collection_worker(args):
    """
    Torch multiprocessing worker for self-play or vs-baseline.
    Returns list of samples from MATCHES.
    """
    mode, model_state, baseline_state, matches_per_worker, device = args

    torch.set_num_threads(1)

    game = BackgammonGame()
    model = get_model().to(device)
    model.load_state_dict(model_state)
    model.eval()

    baseline_model = None
    if baseline_state:
        baseline_model = get_model().to(device)
        baseline_model.load_state_dict(baseline_state)
        baseline_model.eval()

    mcts = MCTS(model, device=device)
    collected = []

    for _ in range(matches_per_worker):
        if mode == "self":
            # UPDATED: Use play_self_play_match
            data, _ = play_self_play_match(game=game, mcts=mcts, model=model, device=device)
        else:
            # UPDATED: Use play_vs_baseline_match
            data, _ = play_vs_baseline_match(
                game=game,
                current_model=model,
                baseline_model=baseline_model,
                mcts_current=mcts,
                device=device
            )
        collected.extend(data)

    return collected

def collect_worker_wrapper(args):
    return collection_worker(args)

def parallel_collect(mode, model, baseline_model, replay_buffer, total_matches, device="cpu"):
    """
    Collect matches in parallel using imap_unordered
    """
    num_workers = mp.cpu_count()
    matches_per_worker = max(1, total_matches // num_workers)

    model_state = model.state_dict()
    baseline_state = baseline_model.state_dict() if baseline_model else None

    ctx = mp.get_context("spawn")

    args_list = [
        (mode, model_state, baseline_state, matches_per_worker, device)
        for _ in range(num_workers)
    ]

    collected = []

    with ctx.Pool(processes=num_workers) as pool:
        pbar = tqdm(total=total_matches, desc=f"Collecting ({mode})", dynamic_ncols=True)
        for result in pool.imap_unordered(collect_worker_wrapper, args_list):
            collected.extend(result)
            # Update progress based on estimated batch size, or just visually
            pbar.update(len(result) // 20 if len(result) > 20 else 1) 
        pbar.close()

    replay_buffer.extend(collected)

# =========================
# TRAIN LOOP
# =========================

def get_strong_opponent(phase, baseline_model, best_model):
    if phase == "vs_baseline":
        return baseline_model
    return best_model

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

    baseline_path = os.path.join(
        os.path.dirname(checkpoint_dir),
        Config.BASELINE_DIR,
        Config.BASELINE_MODEL_NAME
    )
    baseline_config_path = os.path.join(
        os.path.dirname(checkpoint_dir),
        Config.BASELINE_DIR,
        'config.py'
    )

    baseline_model, baseline_elo = None, Config.INITIAL_ELO
    use_baseline = False

    if os.path.exists(baseline_path) and os.path.exists(baseline_config_path):
        baseline_model, baseline_elo = load_model_with_config(
            baseline_config_path,
            baseline_path,
            device
        )
        print(f"✅ Baseline loaded: {baseline_path} (ELO: {baseline_elo:.0f})")
        use_baseline = True
    else:
        print(f"⚠️ No baseline found at {baseline_path}, proceeding with self-play only.")

    replay_buffer = get_replay_buffer(
        Config.BUFFER_SIZE,
        prioritized=True
    )

    game = BackgammonGame()

    if getattr(Config, 'BASELINE_SWITCH_ON_SURPASS', True):
        phase = "vs_baseline" if (use_baseline and current_elo < baseline_elo) else "self_play"
    else:
        phase = "vs_baseline" if use_baseline else "self_play"

    print(f"\n🎮 Training Start: Current ELO={current_elo:.0f}, Best ELO={best_elo:.0f}")
    if use_baseline:
        print(f"   Baseline ELO={baseline_elo:.0f}, Phase={phase}")

    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")

    while train_step < Config.TRAIN_STEPS:
        # -------- COLLECTION PHASE (decoupled) --------
        if train_step % Config.COLLECTION_INTERVAL == 0:
            num_self = int(Config.MATCHES_PER_ITERATION * Config.BASELINE_SELF_PLAY_RATIO)
            num_baseline = Config.MATCHES_PER_ITERATION - num_self

            strong_opponent = get_strong_opponent(phase, baseline_model, best_model)

            if num_self > 0:
                parallel_collect("self", model, None, replay_buffer, num_self)

            if strong_opponent is not None and num_baseline > 0:
                parallel_collect("baseline", model, strong_opponent, replay_buffer, num_baseline)

        # -------- TRAINING PHASE (heavy reuse) --------
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
            pbar.update(1)

            buf_fill = (len(replay_buffer) / Config.BUFFER_SIZE) * 100
            pbar.set_postfix({
                'loss': f'{loss:.2f}',
                'elo': f'{current_elo:.0f}',
                'gnorm': f'{gnorm:.2f}',
                'buf': f'{buf_fill:.1f}%'
            })

            # -------- EVALUATION PHASE --------
            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                print(f"\n\n🎯 ELO Evaluation at Step {train_step} (Phase: {phase})")
                model.eval()
                best_model.eval()

                eval_total = Config.ELO_EVAL_GAMES
                if phase == "vs_baseline":
                    eval_self = int(eval_total * Config.BASELINE_SELF_PLAY_RATIO)
                    eval_base = eval_total - eval_self
                else:
                    eval_self = eval_total
                    eval_base = 0

                wins_total, games_total = 0, 0

                if eval_self > 0:
                    args = (game, model, best_model, eval_self, 'cpu', None)
                    w, t = evaluate_vs_opponent(args)
                    wins_total += w
                    games_total += t
                    print(f"  → vs Best: {w}/{t} wins ({(w/t)*100:.1f}%)")

                if eval_base > 0 and baseline_model is not None:
                    args = (game, model, baseline_model, eval_base, 'cpu', None)
                    w, t = evaluate_vs_opponent(args)
                    wins_total += w
                    games_total += t
                    print(f"  → vs Baseline: {w}/{t} wins ({(w/t)*100:.1f}%)")

                old_elo = current_elo
                if phase == "vs_baseline":
                    opp_elo = (
                        best_elo * Config.BASELINE_SELF_PLAY_RATIO +
                        baseline_elo * (1 - Config.BASELINE_SELF_PLAY_RATIO)
                    )
                else:
                    opp_elo = best_elo

                current_elo = update_elo(current_elo, opp_elo, wins_total, games_total)
                print(f"  📊 Summary: {wins_total}/{games_total} wins | ELO: {old_elo:.0f} -> {current_elo:.0f}")

                if current_elo > best_elo:
                    print(f"  🏆 NEW BEST MODEL (ELO: {current_elo:.0f})")
                    best_elo = current_elo
                    load_model_state_dict(best_model, get_model_state_dict(model))
                    save_checkpoint(model, optimizer, train_step, best_elo, loss, best_path)

                save_checkpoint(model, optimizer, train_step, current_elo, loss, latest_path)

                if (
                    phase == "vs_baseline"
                    and current_elo >= baseline_elo
                    and getattr(Config, 'BASELINE_SWITCH_ON_SURPASS', True)
                ):
                    phase = "self_play"
                    print("  🎯 Surpassed baseline! Switching to 100% self-play.")

                model.train()
                print("-" * 50)

    pbar.close()
    print("\n✅ Training complete!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()