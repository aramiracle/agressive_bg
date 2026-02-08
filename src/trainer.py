import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

from src.config import Config
from src.engine import BackgammonGame
from src.model import get_model
from src.mcts import MCTS
from src.utils.checkpoint import (
    setup_checkpoint_dir, save_checkpoint, load_checkpoint,
    get_model_state_dict, load_model_state_dict
)
from src.utils.elo import evaluate_vs_opponent, update_elo
from src.replay_buffer import get_replay_buffer
from src.utils.game import play_self_play_match
from src.utils.train import train_batch

torch.multiprocessing.set_sharing_strategy("file_system")

def get_cube_epsilon(train_step):
    for i in reversed(range(len(Config.CUBE_CURRICULUM_STAGES))):
        stage = Config.CUBE_CURRICULUM_STAGES[i]
        if train_step >= stage['steps']:
            return stage['epsilon'], stage['cube_weight']
    return Config.CUBE_CURRICULUM_STAGES[0]['epsilon'], Config.CUBE_CURRICULUM_STAGES[0]['cube_weight']

def collection_worker(args):
    model_state, matches_per_worker, device, cube_epsilon = args
    torch.set_num_threads(1)
    
    game = BackgammonGame()
    model = get_model().to(device)
    model.load_state_dict(model_state)
    model.eval()
    mcts = MCTS(model, cpuct=Config.C_PUCT, num_sims=Config.NUM_SIMULATIONS, device=device)

    collected = []
    # Local stats accumulation
    local_stats = {'doubles': 0, 'takes': 0, 'drops': 0, 'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    for _ in range(matches_per_worker):
        game.reset()
        data, _, match_stats = play_self_play_match(
            game, mcts, model, device, 
            is_eval=False, 
            cube_epsilon=cube_epsilon
        )
        collected.extend(data)
        for k in local_stats: local_stats[k] += match_stats[k]

    return collected, local_stats

def collect_worker_wrapper(args):
    return collection_worker(args)

def parallel_collect_self_play(model, replay_buffer, total_matches, device="cpu", cube_epsilon=0.0):
    collection_device = getattr(Config, 'SELF_PLAY_DEVICE', device)
    num_workers = mp.cpu_count()
    matches_per_worker = max(1, total_matches // num_workers)
    
    model_state = model.state_dict()
    ctx = mp.get_context("spawn")
    
    args_list = [(model_state, matches_per_worker, collection_device, cube_epsilon) for _ in range(num_workers)]

    collected_data = []
    agg_stats = {'doubles': 0, 'takes': 0, 'drops': 0, 'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(collect_worker_wrapper, args_list)
        for data, stats in results:
            collected_data.extend(data)
            for k in agg_stats: agg_stats[k] += stats[k]

    replay_buffer.extend(collected_data)
    return agg_stats

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

    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=True, device=device)
    game = BackgammonGame()

    print(f"\n🎮 Training Start: ELO={current_elo:.0f}")
    
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Training")

    while train_step < Config.TRAIN_STEPS:
        cube_epsilon, cube_weight = get_cube_epsilon(train_step)
        
        tqdm.write(f"--- Epoch Phase 1: Collecting {Config.MATCHES_PER_ITERATION} self-play games ---")
        stats = parallel_collect_self_play(model, replay_buffer, Config.MATCHES_PER_ITERATION, device, cube_epsilon)

        if len(replay_buffer) < Config.BATCH_SIZE: continue

        model.train()
        avg_loss = 0
        tqdm.write(f"--- Epoch Phase 2: Training on {Config.TRAIN_UPDATES_PER_ITER} batches ---")
        for update_idx in range(Config.TRAIN_UPDATES_PER_ITER):
            # Print current step within this epoch/iteration
            tqdm.write(f"   -> Batch {update_idx + 1}/{Config.TRAIN_UPDATES_PER_ITER} | Global Step: {train_step + 1}")
            
            Config.CUBE_LOSS_WEIGHT = cube_weight
            loss, gnorm = train_batch(model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler)
            avg_loss += loss
            train_step += 1
            pbar.update(1)
        avg_loss /= Config.TRAIN_UPDATES_PER_ITER

        # Stats display
        n_g, n_d, n_dr = max(1, stats['games']), max(1, stats['doubles']), max(1, stats['drops'])
        pbar.set_postfix({
            'L': f'{avg_loss:.2f}', 'ELO': f'{current_elo:.0f}', 'ε': f'{cube_epsilon:.2f}',
            'Cube': f'D/G:{stats["doubles"]/n_g:.1f} Tk:{stats["takes"]/n_d:.0%} VD:{stats["sum_val_double"]/n_d:.2f}'
        })

        if train_step % Config.ELO_EVAL_INTERVAL == 0:
            tqdm.write(f"--- Epoch Phase 3: Evaluating (Interval Reached) ---")
            model.eval()
            best_model.eval()
            wins, total = evaluate_vs_opponent((game, model, best_model, Config.ELO_EVAL_GAMES, 'cpu', None))
            
            old_elo = current_elo
            current_elo = update_elo(current_elo, best_elo, wins, total)
            tqdm.write(f"   -> Eval Result: {wins}/{total} wins. ELO: {old_elo:.0f}->{current_elo:.0f}. Stats: {stats}")

            if current_elo > best_elo:
                best_elo = current_elo
                load_model_state_dict(best_model, get_model_state_dict(model))
                save_checkpoint(model, optimizer, train_step, best_elo, avg_loss, best_path)
                tqdm.write(f"  --> New Best Model Saved (ELO {best_elo:.0f})")

            save_checkpoint(model, optimizer, train_step, current_elo, avg_loss, latest_path)
            model.train()

    pbar.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()