import os
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
    get_model_state_dict, load_model_state_dict, load_model_with_config
)
from src.utils.elo import evaluate_vs_opponent, update_elo
from src.utils.train import train_batch
from src.utils.game import play_self_play_match, play_vs_baseline_match
from src.replay_buffer import get_replay_buffer

torch.multiprocessing.set_sharing_strategy("file_system")

def get_cube_epsilon(train_step):
    for i in reversed(range(len(Config.CUBE_CURRICULUM_STAGES))):
        stage = Config.CUBE_CURRICULUM_STAGES[i]
        if train_step >= stage['steps']:
            return stage['epsilon'], stage['cube_weight']
    return Config.CUBE_CURRICULUM_STAGES[0]['epsilon'], Config.CUBE_CURRICULUM_STAGES[0]['cube_weight']

def collection_worker(args):
    mode, model_state, baseline_state, matches_per_worker, device, cube_epsilon = args
    torch.set_num_threads(1)

    game = BackgammonGame()
    model = get_model().to(device)
    model.load_state_dict(model_state)
    model.eval()

    mcts_current = MCTS(model, cpuct=Config.C_PUCT, num_sims=Config.NUM_SIMULATIONS, device=device)
    baseline_model = None
    if baseline_state:
        baseline_model = get_model().to(device)
        baseline_model.load_state_dict(baseline_state)
        baseline_model.eval()

    collected = []
    local_stats = {'doubles': 0, 'takes': 0, 'drops': 0, 'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    for _ in range(matches_per_worker):
        game.reset()
        if mode == "self":
            data, _, stats = play_self_play_match(game, mcts_current, model, device, False, cube_epsilon)
        else:
            data, _, stats = play_vs_baseline_match(game, model, baseline_model, mcts_current, device, cube_epsilon)
        
        collected.extend(data)
        for k in local_stats: local_stats[k] += stats[k]

    return collected, local_stats

def collect_worker_wrapper(args): return collection_worker(args)

def parallel_collect(mode, model, baseline_model, replay_buffer, total_matches, device="cpu", cube_epsilon=0.0):
    num_workers = mp.cpu_count()
    matches_per_worker = max(1, total_matches // num_workers)
    model_state = model.state_dict()
    baseline_state = baseline_model.state_dict() if baseline_model else None
    ctx = mp.get_context("spawn")

    args_list = [(mode, model_state, baseline_state, matches_per_worker, device, cube_epsilon) for _ in range(num_workers)]
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

    baseline_path = os.path.join(os.path.dirname(checkpoint_dir), Config.BASELINE_DIR, Config.BASELINE_MODEL_NAME)
    baseline_config_path = os.path.join(os.path.dirname(checkpoint_dir), Config.BASELINE_DIR, 'config.py')
    
    baseline_model, baseline_elo = None, Config.INITIAL_ELO
    use_baseline = False
    if os.path.exists(baseline_path):
        baseline_model, baseline_elo = load_model_with_config(baseline_config_path, baseline_path, device)
        use_baseline = True

    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=True, device=device)
    phase = "vs_baseline" if (use_baseline and current_elo < baseline_elo) else "self_play"

    print(f"\n🎮 Start vs Baseline: CurELO={current_elo:.0f} BaseELO={baseline_elo:.0f}")
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Training")

    while train_step < Config.TRAIN_STEPS:
        cube_epsilon, cube_weight = get_cube_epsilon(train_step)
        num_self = int(Config.MATCHES_PER_ITERATION * Config.BASELINE_SELF_PLAY_RATIO)
        num_baseline = Config.MATCHES_PER_ITERATION - num_self
        opponent = baseline_model if phase == "vs_baseline" else best_model

        stats_self = {'doubles':0, 'takes':0, 'drops':0, 'sum_val_double':0, 'sum_val_drop':0, 'games':0}
        stats_base = {'doubles':0, 'takes':0, 'drops':0, 'sum_val_double':0, 'sum_val_drop':0, 'games':0}

        if num_self > 0:
            tqdm.write(f"--- Epoch Phase: Collecting {num_self} self-play games ---")
            stats_self = parallel_collect("self", model, None, replay_buffer, num_self, device, cube_epsilon)
        
        if opponent and num_baseline > 0:
            tqdm.write(f"--- Epoch Phase: Collecting {num_baseline} baseline games ---")
            stats_base = parallel_collect("baseline", model, opponent, replay_buffer, num_baseline, device, cube_epsilon)

        if len(replay_buffer) < Config.BATCH_SIZE: continue

        model.train()
        avg_loss = 0
        tqdm.write(f"--- Epoch Phase: Training on {Config.TRAIN_UPDATES_PER_ITER} batches ---")
        for ـ in range(Config.TRAIN_UPDATES_PER_ITER):
            Config.CUBE_LOSS_WEIGHT = cube_weight
            loss, gnorm = train_batch(model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler)
            avg_loss += loss
            train_step += 1
            pbar.update(1)
        avg_loss /= Config.TRAIN_UPDATES_PER_ITER

        # Combine stats for display
        s_d = stats_self['doubles'] + stats_base['doubles']
        s_g = stats_self['games'] + stats_base['games']
        s_t = stats_self['takes'] + stats_base['takes']
        s_dr = stats_self['drops'] + stats_base['drops']
        s_vd = stats_self['sum_val_double'] + stats_base['sum_val_double']
        
        n_d, n_g = max(1, s_d), max(1, s_g)

        pbar.set_postfix({
            'L': f'{avg_loss:.2f}', 'ELO': f'{current_elo:.0f}',
            'Cube': f'D/G:{s_d/n_g:.1f} Tk:{s_t/n_d:.0%} VD:{s_vd/n_d:.2f}'
        })

        if train_step % Config.ELO_EVAL_INTERVAL == 0:
            tqdm.write(f"--- Epoch Phase: Evaluating (Interval Reached) ---")
            model.eval()
            best_model.eval()
            
            wins, total = evaluate_vs_opponent((BackgammonGame(), model, best_model, Config.ELO_EVAL_GAMES, 'cpu', None))
            current_elo = update_elo(current_elo, best_elo, wins, total)
            
            tqdm.write(f"   -> Eval vs Best: {wins}/{total}. ELO: {current_elo:.0f}")
            if current_elo > best_elo:
                best_elo = current_elo
                load_model_state_dict(best_model, get_model_state_dict(model))
                save_checkpoint(model, optimizer, train_step, best_elo, avg_loss, best_path)
                tqdm.write(f"  --> New Best Model Saved (ELO {best_elo:.0f})")

            if phase == "vs_baseline" and current_elo >= baseline_elo:
                phase = "self_play"
                tqdm.write(">>> Surpassed Baseline! Switching to Self-Play.")

            save_checkpoint(model, optimizer, train_step, current_elo, avg_loss, latest_path)
            model.train()

    pbar.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()