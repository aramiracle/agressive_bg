import ray
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from tqdm import tqdm

from config import Config
from bg_engine import BackgammonGame
from model import get_model
from checkpoint import save_checkpoint, setup_checkpoint_dir
from elo import update_elo, evaluate_vs_opponent
from worker import SelfPlayWorker

def print_banner(elo, checkpoint_dir, model):
    """Print training configuration banner."""
    param_count = model.count_parameters() if hasattr(model, 'count_parameters') else 0
    param_str = f"{param_count:,}" if param_count < 1_000_000 else f"{param_count/1_000_000:.2f}M"
    
    print("=" * 70)
    print("🎲 BACKGAMMON AI TRAINER")
    print("=" * 70)
    print(f"   • Device: {Config.DEVICE}")
    print(f"   • Workers: {Config.NUM_WORKERS} (on {Config.WORKER_DEVICE})")
    print(f"   • Batch Size: {Config.BATCH_SIZE}")
    print("-" * 70)
    print(f"🧠 Model Params: {param_str}")
    print("=" * 70)
    print(f"🏁 Initial ELO: {elo}")
    print("=" * 70)
    print()

def prepare_batch(replay_buffer, batch_size):
    # Random sampling
    batch = random.sample(replay_buffer, batch_size)
    
    # Batch is list of tuples: (b_t, c_t, (s, e), reward)
    # b_t and c_t are already Tensors on CPU.
    
    # 1. Stack Tensors (Fastest way to combine existing tensors)
    b_states = torch.stack([x[0] for x in batch]).to(Config.DEVICE)
    contexts = torch.stack([x[1] for x in batch]).to(Config.DEVICE)
    
    # 2. Process Actions (Integers -> Tensor)
    actions = [x[2] for x in batch]
    target_from = torch.tensor([a[0] for a in actions], dtype=torch.long, device=Config.DEVICE)
    target_to = torch.tensor([a[1] for a in actions], dtype=torch.long, device=Config.DEVICE)
    
    # 3. Rewards
    rewards = torch.tensor([x[3] for x in batch], dtype=torch.float, device=Config.DEVICE)
    
    return b_states, contexts, target_from, target_to, rewards

def train_step(model, optimizer, b_states, contexts, target_from, target_to, rewards):
    # Automatic Mixed Precision for speed if on CUDA
    use_amp = (Config.DEVICE == 'cuda')
    
    with torch.amp.autocast('cuda', enabled=use_amp):
        p_from, p_to, v, _ = model(b_states, contexts)
        
        loss_v = nn.MSELoss()(v.squeeze(), rewards)
        loss_p = nn.CrossEntropyLoss()(p_from, target_from) + nn.CrossEntropyLoss()(p_to, target_to)
        total_loss = loss_v + loss_p
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def sync_workers(model, workers):
    # Move state dict to CPU for serialization to workers
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    # Fire and forget update
    [w.update_model.remote(state_dict) for w in workers]

def run_elo_evaluation(step, model, older_model, eval_game, elo, older_elo, avg_loss):
    tqdm.write("")
    tqdm.write(f"🎯 ELO EVALUATION @ Step {step} | Loss: {avg_loss:.4f}")
    
    model.eval()
    # evaluate_vs_opponent handles the MCTS creation internally if not passed, 
    # but our optimized elo.py (implied) might need the model instance.
    win_rate = evaluate_vs_opponent(eval_game, model, older_model, show_progress=True)
    model.train()
    
    new_elo = update_elo(elo, older_elo, win_rate)
    elo_change = new_elo - elo
    
    wins = int(win_rate * Config.ELO_EVAL_GAMES)
    tqdm.write(f"   Result: {wins}/{Config.ELO_EVAL_GAMES} wins ({win_rate*100:.0f}%)")
    tqdm.write(f"   ELO: {elo:.0f} → {new_elo:.0f} ({elo_change:+.0f})")
    
    return new_elo

def train():
    ray.init(include_dashboard=False, logging_level="ERROR")
    checkpoint_dir, best_model_path, latest_model_path = setup_checkpoint_dir()
    
    # Setup Model
    raw_model = get_model().to(Config.DEVICE)
    
    # Optimization: Compile model (PyTorch 2.0+)
    try:
        # Note: 'reduce-overhead' is good for small batches/RL loops
        model = torch.compile(raw_model) 
        print("✅ Model compiled with torch.compile")
    except Exception as e:
        model = raw_model
        print(f"⚠️ torch.compile not available/failed, using eager mode. ({e})")

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # Initial State for Workers (Send CPU state)
    cpu_state = {k: v.cpu() for k, v in raw_model.state_dict().items()}
    workers = [SelfPlayWorker.remote(cpu_state) for _ in range(Config.NUM_WORKERS)]
    
    replay_buffer = deque(maxlen=Config.BUFFER_SIZE)
    elo = Config.INITIAL_ELO
    best_elo = Config.INITIAL_ELO
    
    # For Eval (Comparison against previous version)
    eval_game = BackgammonGame()
    older_model = get_model().to(Config.DEVICE)
    older_model.load_state_dict(raw_model.state_dict())
    older_model.eval()
    older_elo = Config.INITIAL_ELO
    
    total_matches_played = 0
    total_samples_collected = 0
    loss_history = []
    
    print_banner(elo, checkpoint_dir, raw_model)
    
    pbar = tqdm(range(Config.TRAIN_STEPS), desc="Training")
    
    for step in pbar:
        # Collect Data from Workers
        results = ray.get([w.play_match.remote() for w in workers])
        
        step_samples = 0
        for game_data, scores in results:
            replay_buffer.extend(game_data)
            step_samples += len(game_data)
        
        total_matches_played += len(results)
        total_samples_collected += step_samples
        
        # Train
        if len(replay_buffer) > Config.BATCH_SIZE:
            batch_data = prepare_batch(replay_buffer, Config.BATCH_SIZE)
            loss = train_step(model, optimizer, *batch_data)
            
            loss_history.append(loss)
            avg_loss = sum(loss_history[-Config.LOSS_AVG_WINDOW:]) / min(len(loss_history), Config.LOSS_AVG_WINDOW)
            
            # Sync workers periodically (every 10 steps to save overhead)
            if step % 10 == 0:
                sync_workers(raw_model, workers)
            
            pbar.set_postfix({'Loss': f'{avg_loss:.3f}', 'ELO': f'{elo:.0f}'})
            
            # --- ELO EVALUATION & SAVING LOGIC ---
            if step > 0 and step % Config.ELO_EVAL_INTERVAL == 0:
                elo = run_elo_evaluation(
                    step, raw_model, older_model, eval_game,
                    elo, older_elo, avg_loss
                )
                
                if elo > best_elo:
                    best_elo = elo
                    # Save best model
                    save_checkpoint(raw_model, optimizer, step, elo, avg_loss, best_model_path)
                    tqdm.write(f"   💾 New best model saved!")
                
                # Update "older_model" to current state for next comparison
                older_model.load_state_dict(raw_model.state_dict())
                older_elo = elo
            
            if step > 0 and step % Config.SAVE_INTERVAL == 0:
                save_checkpoint(raw_model, optimizer, step, elo, avg_loss, latest_model_path)
    
    # Final Save
    final_loss = np.mean(loss_history) if loss_history else 0
    save_checkpoint(raw_model, optimizer, Config.TRAIN_STEPS, elo, final_loss, latest_model_path)
    
    print()
    print("=" * 70)
    print("🏆 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   📊 Final ELO: {elo:.0f}")
    print(f"   📊 Best ELO: {best_elo:.0f}")
    ray.shutdown()

if __name__ == "__main__":
    train()