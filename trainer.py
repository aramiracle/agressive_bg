"""Main training loop for backgammon AI."""

import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm

from config import Config
from bg_engine import BackgammonGame
from model import BackgammonTransformer
from checkpoint import save_checkpoint, setup_checkpoint_dir
from elo import update_elo, evaluate_vs_opponent
from worker import SelfPlayWorker


def print_banner(elo, checkpoint_dir, model):
    """Print training configuration banner."""
    param_count = model.count_parameters()
    param_str = f"{param_count:,}" if param_count < 1_000_000 else f"{param_count/1_000_000:.2f}M"
    
    print("=" * 70)
    print("🎲 BACKGAMMON AI TRAINER")
    print("=" * 70)
    print("📊 Config:")
    print(f"   • Device: {Config.DEVICE}")
    print(f"   • Workers: {Config.NUM_WORKERS}")
    print(f"   • Batch Size: {Config.BATCH_SIZE}")
    print(f"   • Buffer Size: {Config.BUFFER_SIZE}")
    print(f"   • MCTS Simulations: {Config.NUM_SIMULATIONS}")
    print(f"   • Training Steps: {Config.TRAIN_STEPS}")
    print(f"   • ELO Eval Interval: {Config.ELO_EVAL_INTERVAL} steps")
    print(f"   • Checkpoint Dir: {checkpoint_dir.absolute()}")
    print("-" * 70)
    print("🧠 Model:")
    print(f"   • Parameters: {param_str}")
    print(f"   • D_MODEL: {Config.D_MODEL}, Heads: {Config.N_HEAD}, Layers: {Config.N_LAYERS}")
    print(f"   • Feedforward: {Config.DIM_FEEDFORWARD}, Value Hidden: {Config.VALUE_HIDDEN}")
    print("=" * 70)
    print(f"🏁 Initial ELO: {elo}")
    print("=" * 70)
    print()


def print_final_summary(elo, best_elo, total_matches, total_samples, final_loss, checkpoint_dir):
    """Print final training summary."""
    print()
    print("=" * 70)
    print("🏆 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   📊 Final ELO: {elo:.0f}")
    print(f"   📊 Best ELO: {best_elo:.0f}")
    print(f"   📈 Total Matches: {total_matches}")
    print(f"   📉 Total Samples: {total_samples}")
    print(f"   📉 Final Avg Loss: {final_loss:.4f}")
    print("-" * 70)
    print(f"   💾 Models saved to: {checkpoint_dir.absolute()}")
    print(f"      • best_model.pt  (ELO: {best_elo:.0f})")
    print(f"      • latest_model.pt (ELO: {elo:.0f})")
    print("=" * 70)


def collect_self_play_data(workers):
    """Collect training data from all workers."""
    futures = [w.play_match.remote() for w in workers]
    results = ray.get(futures)
    return results


def prepare_batch(replay_buffer, batch_size):
    """Prepare a training batch from replay buffer."""
    batch_indices = np.random.choice(len(replay_buffer), batch_size)
    
    b_states = torch.from_numpy(
        np.array([replay_buffer[i][0] for i in batch_indices])
    ).long().to(Config.DEVICE)
    
    contexts = torch.from_numpy(
        np.array([replay_buffer[i][1] for i in batch_indices])
    ).float().to(Config.DEVICE)
    
    actions = [replay_buffer[i][2] for i in batch_indices]
    target_from = torch.LongTensor([a[0] for a in actions]).to(Config.DEVICE)
    target_to = torch.LongTensor([a[1] for a in actions]).to(Config.DEVICE)
    
    rewards = torch.FloatTensor(
        [replay_buffer[i][3] for i in batch_indices]
    ).to(Config.DEVICE)
    
    return b_states, contexts, target_from, target_to, rewards


def train_step(model, optimizer, b_states, contexts, target_from, target_to, rewards):
    """Perform a single training step."""
    p_from, p_to, v, _ = model(b_states, contexts)
    
    loss_v = nn.MSELoss()(v.squeeze(), rewards)
    loss_p = nn.CrossEntropyLoss()(p_from, target_from) + nn.CrossEntropyLoss()(p_to, target_to)
    total_loss = loss_v + loss_p
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()


def sync_workers(model, workers):
    """Synchronize model weights to all workers."""
    state_dict = model.cpu().state_dict()
    for w in workers:
        w.update_model.remote(state_dict)
    model.to(Config.DEVICE)


def run_elo_evaluation(step, model, older_model, eval_game, elo, older_elo, 
                       avg_loss, total_matches, total_samples):
    """Run ELO evaluation and return updated ELO values."""
    tqdm.write("")
    tqdm.write("=" * 70)
    tqdm.write(f"🎯 ELO EVALUATION @ Step {step}")
    tqdm.write(f"   📈 Stats: {total_matches} matches | {total_samples} samples")
    tqdm.write(f"   📉 Avg Loss (last {Config.LOSS_AVG_WINDOW}): {avg_loss:.4f}")
    tqdm.write("")
    
    model.eval()
    win_rate = evaluate_vs_opponent(eval_game, model, older_model, show_progress=True)
    model.train()
    
    old_elo_value = elo
    new_elo = update_elo(elo, older_elo, win_rate)
    
    elo_change = new_elo - old_elo_value
    if elo_change > 0:
        elo_symbol = "⬆️ "
        elo_status = "improved"
    elif elo_change < 0:
        elo_symbol = "⬇️ "
        elo_status = "declined"
    else:
        elo_symbol = "➡️ "
        elo_status = "stable"
    
    wins = int(win_rate * Config.ELO_EVAL_GAMES)
    tqdm.write(f"   🎲 Results vs Older Model: {wins}/{Config.ELO_EVAL_GAMES} wins ({win_rate*100:.0f}%)")
    tqdm.write(f"   {elo_symbol}ELO: {old_elo_value:.0f} → {new_elo:.0f} ({elo_change:+.0f}) [{elo_status}]")
    
    return new_elo


def train():
    """Main training loop."""
    # Initialize Ray
    ray.init(
        include_dashboard=False,
        _metrics_export_port=None,
        logging_level="WARNING"
    )
    
    # Setup checkpoints
    checkpoint_dir, best_model_path, latest_model_path = setup_checkpoint_dir()
    
    # Initialize model and optimizer
    model = BackgammonTransformer().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # Initialize workers
    workers = [SelfPlayWorker.remote(model.cpu().state_dict()) for _ in range(Config.NUM_WORKERS)]
    model.to(Config.DEVICE)
    
    # Initialize training state
    replay_buffer = deque(maxlen=Config.BUFFER_SIZE)
    elo = Config.INITIAL_ELO
    best_elo = Config.INITIAL_ELO
    
    # Opponent model for ELO evaluation
    eval_game = BackgammonGame()
    older_model = BackgammonTransformer().to(Config.DEVICE)
    older_model.load_state_dict(model.state_dict())
    older_model.eval()
    older_elo = Config.INITIAL_ELO
    
    # Statistics
    total_matches_played = 0
    total_samples_collected = 0
    loss_history = []
    
    print_banner(elo, checkpoint_dir, model)
    
    # Main training loop
    pbar = tqdm(range(Config.TRAIN_STEPS), desc="Training", unit="step")
    
    for step in pbar:
        # 1. Self-play data collection
        results = collect_self_play_data(workers)
        
        step_samples = 0
        for game_data, scores in results:
            replay_buffer.extend(game_data)
            step_samples += len(game_data)
        
        total_matches_played += len(results)
        total_samples_collected += step_samples
        
        # 2. Training
        if len(replay_buffer) > Config.BATCH_SIZE:
            batch_data = prepare_batch(replay_buffer, Config.BATCH_SIZE)
            loss = train_step(model, optimizer, *batch_data)
            
            loss_history.append(loss)
            avg_loss = np.mean(loss_history[-Config.LOSS_AVG_WINDOW:])
            
            # Sync to workers
            sync_workers(model, workers)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss:.3f}',
                'AvgLoss': f'{avg_loss:.3f}',
                'ELO': f'{elo:.0f}',
                'Buffer': f'{len(replay_buffer)}'
            })
            
            # 3. ELO Evaluation
            if step > 0 and step % Config.ELO_EVAL_INTERVAL == 0:
                elo = run_elo_evaluation(
                    step, model, older_model, eval_game,
                    elo, older_elo, avg_loss,
                    total_matches_played, total_samples_collected
                )
                
                # Save best model
                if elo > best_elo:
                    best_elo = elo
                    save_checkpoint(model, optimizer, step, elo, avg_loss, best_model_path)
                    tqdm.write(f"   💾 New best model saved! (ELO: {elo:.0f})")
                
                tqdm.write("=" * 70)
                tqdm.write("")
                
                # Update older model
                older_model.load_state_dict(model.state_dict())
                older_elo = elo
            
            # 4. Periodic checkpoint
            if step > 0 and step % Config.SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, step, elo, avg_loss, latest_model_path)
    
    # Final save
    final_loss = np.mean(loss_history[-Config.LOSS_AVG_WINDOW:]) if loss_history else 0
    save_checkpoint(model, optimizer, Config.TRAIN_STEPS, elo, final_loss, latest_model_path)
    
    print_final_summary(
        elo, best_elo, total_matches_played, 
        total_samples_collected, final_loss, checkpoint_dir
    )
    
    ray.shutdown()


if __name__ == "__main__":
    train()
