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
from src.utils.elo import evaluate_combined, update_elo
from src.utils.train import train_batch
from src.utils.game import play_self_play_match, play_vs_baseline_match
from src.replay_buffer import get_replay_buffer
from src.utils.match_equity import MatchEquityTable

torch.multiprocessing.set_sharing_strategy("file_system")


def get_cube_epsilon(train_step):
    for i in reversed(range(len(Config.CUBE_CURRICULUM_STAGES))):
        stage = Config.CUBE_CURRICULUM_STAGES[i]
        if train_step >= stage['steps']:
            return stage['epsilon'], stage['cube_weight']
    return Config.CUBE_CURRICULUM_STAGES[0]['epsilon'], Config.CUBE_CURRICULUM_STAGES[0]['cube_weight']


def collection_worker(args):
    mode, model_state, baseline_state, equity_table_state, matches_per_worker, device, cube_epsilon = args
    torch.set_num_threads(1)

    game  = BackgammonGame()
    model = get_model().to(device)
    model.load_state_dict(model_state)
    model.eval()

    mcts_current = MCTS(model, cpuct=Config.C_PUCT, num_sims=Config.NUM_SIMULATIONS, device=device)

    baseline_model = None
    if baseline_state:
        baseline_model = get_model().to(device)
        baseline_model.load_state_dict(baseline_state)
        baseline_model.eval()

    equity_table = MatchEquityTable()
    equity_table.equity_table = equity_table_state

    collected   = []
    local_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                   'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    for _ in range(matches_per_worker):
        game.reset()
        if mode == "self":
            data, _, stats = play_self_play_match(
                game, mcts_current, model, device, equity_table, False, cube_epsilon
            )
        else:
            data, _, stats = play_vs_baseline_match(
                game, model, baseline_model, mcts_current, device, equity_table, cube_epsilon
            )

        collected.extend(data)
        for k in local_stats:
            local_stats[k] += stats[k]

    return collected, local_stats, equity_table.equity_table


def collect_worker_wrapper(args):
    return collection_worker(args)


def parallel_collect(mode, model, baseline_model, equity_table, replay_buffer,
                     total_matches, device="cpu", cube_epsilon=0.0):
    num_workers        = mp.cpu_count()
    matches_per_worker = max(1, total_matches // num_workers)
    model_state        = model.state_dict()
    baseline_state     = baseline_model.state_dict() if baseline_model else None
    equity_table_state = equity_table.equity_table.copy()
    ctx                = mp.get_context("spawn")

    args_list = [
        (mode, model_state, baseline_state, equity_table_state, matches_per_worker, device, cube_epsilon)
        for _ in range(num_workers)
    ]
    collected_data = []
    agg_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                 'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}
    worker_equity_tables = []

    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(collect_worker_wrapper, args_list)
        for data, stats, worker_equity in results:
            collected_data.extend(data)
            for k in agg_stats:
                agg_stats[k] += stats[k]
            worker_equity_tables.append(worker_equity)

    # Merge equity tables
    for key in equity_table.equity_table.keys():
        values = [wt.get(key, equity_table.equity_table[key]) for wt in worker_equity_tables]
        equity_table.equity_table[key] = sum(values) / len(values)

    replay_buffer.extend(collected_data)
    return agg_stats


def train():
    checkpoint_dir, best_path, latest_path = setup_checkpoint_dir()
    device = torch.device(Config.DEVICE)

    model      = get_model().to(device)
    best_model = get_model().to(device)
    optimizer  = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scaler     = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    cp_latest   = load_checkpoint(latest_path, model, optimizer, device)
    train_step  = cp_latest['step'] if cp_latest else 0
    current_elo = cp_latest['elo']  if cp_latest else Config.INITIAL_ELO

    cp_best  = load_checkpoint(best_path, best_model, None, device)
    best_elo = cp_best['elo'] if cp_best else current_elo

    # Initialize match equity table
    equity_table = MatchEquityTable(match_target=Config.MATCH_TARGET, learning_rate=0.01)
    equity_path = os.path.join(checkpoint_dir, 'match_equity.pt')
    if os.path.exists(equity_path):
        try:
            equity_table.load(equity_path)
            tqdm.write(f"   Loaded match equity table")
        except:
            tqdm.write(f"   Using fresh equity table")

    # Load external baseline
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

    baseline_model = None
    baseline_elo   = Config.INITIAL_ELO
    use_baseline   = False

    if os.path.exists(baseline_path):
        baseline_model, baseline_elo = load_model_with_config(
            baseline_config_path, baseline_path, device
        )
        use_baseline = True

    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=True, device=device)
    phase = "vs_baseline" if (use_baseline and current_elo < baseline_elo) else "self_play"

    print(
        f"\n🎮 Start vs Baseline: CurELO={current_elo:.0f} "
        f"BaseELO={baseline_elo:.0f}"
    )
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Training")

    while train_step < Config.TRAIN_STEPS:
        cube_epsilon, cube_weight = get_cube_epsilon(train_step)

        num_self     = int(Config.MATCHES_PER_ITERATION * Config.BASELINE_SELF_PLAY_RATIO)
        num_opponent = Config.MATCHES_PER_ITERATION - num_self
        opponent     = baseline_model if phase == "vs_baseline" else best_model

        stats_self = {'doubles': 0, 'takes': 0, 'drops': 0,
                      'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}
        stats_opp  = {'doubles': 0, 'takes': 0, 'drops': 0,
                      'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

        if num_self > 0:
            tqdm.write(f"--- Epoch Phase: Collecting {num_self} self-play games ---")
            stats_self = parallel_collect(
                "self", model, None, equity_table, replay_buffer, num_self, device, cube_epsilon
            )

        if opponent is not None and num_opponent > 0:
            tqdm.write(f"--- Epoch Phase: Collecting {num_opponent} opponent games ---")
            stats_opp = parallel_collect(
                "baseline", model, opponent, equity_table, replay_buffer, num_opponent, device, cube_epsilon
            )

        if len(replay_buffer) < Config.BATCH_SIZE:
            continue

        model.train()
        avg_loss = 0.0
        tqdm.write(f"--- Epoch Phase: Training on {Config.TRAIN_UPDATES_PER_ITER} batches ---")
        for _ in range(Config.TRAIN_UPDATES_PER_ITER):
            Config.CUBE_LOSS_WEIGHT = cube_weight
            loss, gnorm = train_batch(
                model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler
            )
            avg_loss   += loss
            train_step += 1
            pbar.update(1)
        avg_loss /= Config.TRAIN_UPDATES_PER_ITER

        s_d  = stats_self['doubles']        + stats_opp['doubles']
        s_g  = stats_self['games']          + stats_opp['games']
        s_t  = stats_self['takes']          + stats_opp['takes']
        s_vd = stats_self['sum_val_double'] + stats_opp['sum_val_double']
        n_d  = max(1, s_d)
        n_g  = max(1, s_g)

        pbar.set_postfix({
            'L':    f'{avg_loss:.2f}',
            'ELO':  f'{current_elo:.0f}',
            'Cube': f'D/G:{s_d/n_g:.1f} Tk:{s_t/n_d:.0%} VD:{s_vd/n_d:.2f}'
        })

        if train_step % Config.ELO_EVAL_INTERVAL == 0:
            tqdm.write("--- Epoch Phase: Evaluating (Interval Reached) ---")
            model.eval()
            best_model.eval()

            eval_baseline_model = baseline_model if phase == "vs_baseline" else None
            eval_baseline_elo   = baseline_elo   if phase == "vs_baseline" else best_elo

            total_wins, total_games, opponent_elo = evaluate_combined(
                model          = model,
                best_model     = best_model,
                baseline_model = eval_baseline_model,
                best_elo       = best_elo,
                baseline_elo   = eval_baseline_elo,
                total_games    = Config.ELO_EVAL_GAMES,
                device         = 'cpu',
            )

            old_elo     = current_elo
            current_elo = update_elo(current_elo, opponent_elo, total_wins, total_games)
            tqdm.write(
                f"   -> Eval: {int(total_wins)}/{total_games} wins | "
                f"opp_elo={opponent_elo:.0f} | "
                f"ELO: {old_elo:.0f} -> {current_elo:.0f}"
            )

            if current_elo > best_elo:
                best_elo = current_elo
                load_model_state_dict(best_model, get_model_state_dict(model))
                save_checkpoint(model, optimizer, train_step, best_elo, avg_loss, best_path)
                tqdm.write(f"  --> New Best Model Saved (ELO {best_elo:.0f})")

            if phase == "vs_baseline" and current_elo >= baseline_elo:
                phase = "self_play"
                tqdm.write(">>> Surpassed Baseline! Switching to Self-Play for collection AND evaluation.")

            save_checkpoint(model, optimizer, train_step, current_elo, avg_loss, latest_path)
            equity_table.save(equity_path)
            
            equity_table.print_table()
            
            model.train()

    pbar.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()