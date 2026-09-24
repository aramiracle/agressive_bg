import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

from src.config import Config
from src.engine import BackgammonGame
from src.model import get_model
from src.search import Searcher
from src.utils.checkpoint import (
    setup_checkpoint_dir, save_checkpoint, load_checkpoint,
    get_model_state_dict, load_model_state_dict, warm_start,
    load_model_with_config, baseline_artifact_paths,
)
from src.utils.elo import (
    evaluate_combined, update_elo, passes_gate, promoted_elo, plays_against_baseline,
)
from src.replay_buffer import get_replay_buffer
from src.utils.game import play_self_play_match
from src.utils.train import train_batch
from src.trainer_vs_baseline import parallel_collect, split_matches
from src.utils.match_equity import MatchEquityTable

torch.multiprocessing.set_sharing_strategy("file_system")


def get_cube_epsilon(train_step):
    for i in reversed(range(len(Config.CUBE_CURRICULUM_STAGES))):
        stage = Config.CUBE_CURRICULUM_STAGES[i]
        if train_step >= stage['steps']:
            return stage['epsilon'], stage['cube_weight']
    return Config.CUBE_CURRICULUM_STAGES[0]['epsilon'], Config.CUBE_CURRICULUM_STAGES[0]['cube_weight']


def collection_worker(args):
    model_state, equity_table_state, matches_per_worker, device, cube_epsilon = args
    torch.set_num_threads(1)

    game  = BackgammonGame()
    model = get_model().to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Reconstruct match equity table in worker
    equity_table = MatchEquityTable()
    equity_table.equity_table = equity_table_state

    searcher = Searcher(model, device=device, equity_table=equity_table)

    collected   = []
    local_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                   'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    for _ in range(matches_per_worker):
        game.reset()
        data, _, match_stats = play_self_play_match(
            game, searcher, model, device, equity_table,
            is_eval=False,
            cube_epsilon=cube_epsilon
        )
        collected.extend(data)
        for k in local_stats:
            local_stats[k] += match_stats[k]

    return collected, local_stats, equity_table.pop_observations()


def collect_worker_wrapper(args):
    return collection_worker(args)


def parallel_collect_self_play(model, equity_table, replay_buffer, total_matches, device="cpu", cube_epsilon=0.0):
    collection_device = getattr(Config, 'SELF_PLAY_DEVICE', device)
    counts = split_matches(total_matches, mp.cpu_count())

    model_state = model.state_dict()
    equity_table_state = equity_table.equity_table.copy()  # Send current table to workers
    ctx         = mp.get_context("spawn")

    args_list = [
        (model_state, equity_table_state, count, collection_device, cube_epsilon)
        for count in counts
    ]

    collected_data = []
    agg_stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                 'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}

    if args_list:
        with ctx.Pool(processes=len(args_list)) as pool:
            results = pool.map(collect_worker_wrapper, args_list)
            for data, stats, observations in results:
                collected_data.extend(data)
                for k in agg_stats:
                    agg_stats[k] += stats[k]
                for scores_seen, i_won in observations:
                    equity_table.update_from_match(scores_seen, i_won, record=False)

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

    if cp_latest is None and warm_start(model, Config.INIT_FROM, device):
        tqdm.write(f"   Warm-started weights from {Config.INIT_FROM}")

    cp_best  = load_checkpoint(best_path, best_model, None, device)
    best_elo = cp_best['elo'] if cp_best else current_elo
    if cp_best is None:
        # Fresh run: the gate opponent starts as a copy of the current weights.
        load_model_state_dict(best_model, get_model_state_dict(model))

    # Initialize match equity table
    equity_table = MatchEquityTable(match_target=Config.MATCH_TARGET, learning_rate=0.01)
    equity_path = os.path.join(checkpoint_dir, 'match_equity.pt')
    if os.path.exists(equity_path):
        try:
            equity_table.load(equity_path)
            tqdm.write(f"   Loaded match equity table from {equity_path}")
        except:
            tqdm.write(f"   Failed to load equity table, using fresh initialization")

    # Frozen baseline: eval mixes it in while best is weaker, and training
    # games do too while either rating is still under it.
    baseline_model = None
    baseline_equity_table = None
    baseline_elo   = Config.INITIAL_ELO
    baseline_path, baseline_config_path, baseline_equity_path = baseline_artifact_paths()
    if os.path.exists(baseline_path):
        try:
            baseline_model, baseline_elo = load_model_with_config(
                baseline_config_path, baseline_path, device
            )
            tqdm.write(f"   Baseline loaded: ELO {baseline_elo:.0f}")
            baseline_equity_table = MatchEquityTable(
                match_target=Config.MATCH_TARGET, learning_rate=0.01
            )
            if os.path.exists(baseline_equity_path):
                try:
                    baseline_equity_table.load(baseline_equity_path)
                    tqdm.write("   Loaded baseline match equity table")
                except Exception:
                    tqdm.write("   Using fresh equity table for baseline")
        except Exception as e:
            tqdm.write(f"   Baseline load failed ({e}), self-play and eval vs best only.")
            baseline_model = None
            baseline_equity_table = None
            baseline_elo   = Config.INITIAL_ELO

    replay_buffer = get_replay_buffer(Config.BUFFER_SIZE, prioritized=True, device=device)

    print(
        f"\n🎮 Training Start: stage={Config.STAGE} "
        f"target={Config.MATCH_TARGET} cube={Config.CUBE_ENABLED} "
        f"ELO={current_elo:.0f}"
    )
    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Training")

    while train_step < Config.TRAIN_STEPS:
        cube_epsilon, cube_weight = get_cube_epsilon(train_step)

        play_baseline = plays_against_baseline(
            baseline_model is not None, current_elo, best_elo, baseline_elo,
        )
        if play_baseline:
            num_self = int(Config.MATCHES_PER_ITERATION * Config.BASELINE_SELF_PLAY_RATIO)
            num_base = Config.MATCHES_PER_ITERATION - num_self
            tqdm.write(
                f"--- Epoch Phase 1: Collecting {num_self} self-play "
                f"+ {num_base} vs baseline ---"
            )
            stats = {'doubles': 0, 'takes': 0, 'drops': 0,
                     'sum_val_double': 0.0, 'sum_val_drop': 0.0, 'games': 0}
            if num_self > 0:
                stats_self = parallel_collect_self_play(
                    model, equity_table, replay_buffer, num_self, device, cube_epsilon
                )
                for key in stats:
                    stats[key] += stats_self[key]
            if num_base > 0:
                stats_base = parallel_collect(
                    "baseline", model, baseline_model, equity_table, replay_buffer,
                    num_base, Config.SELF_PLAY_DEVICE, cube_epsilon,
                    baseline_config_path=baseline_config_path,
                    baseline_equity_table=baseline_equity_table,
                )
                for key in stats:
                    stats[key] += stats_base[key]
        else:
            tqdm.write(f"--- Epoch Phase 1: Collecting {Config.MATCHES_PER_ITERATION} self-play games ---")
            stats = parallel_collect_self_play(
                model, equity_table, replay_buffer, Config.MATCHES_PER_ITERATION, device, cube_epsilon
            )

        if len(replay_buffer) < Config.BATCH_SIZE:
            continue

        model.train()
        avg_loss = 0.0
        tqdm.write(f"--- Epoch Phase 2: Training on {Config.TRAIN_UPDATES_PER_ITER} batches ---")
        for update_idx in range(Config.TRAIN_UPDATES_PER_ITER):
            tqdm.write(
                f"   -> Batch {update_idx + 1}/{Config.TRAIN_UPDATES_PER_ITER} "
                f"| Global Step: {train_step + 1}"
            )
            Config.CUBE_LOSS_WEIGHT = cube_weight
            loss, gnorm = train_batch(
                model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler
            )
            avg_loss   += loss
            train_step += 1
            pbar.update(1)
        avg_loss /= Config.TRAIN_UPDATES_PER_ITER

        n_g = max(1, stats['games'])
        n_d = max(1, stats['doubles'])
        pbar.set_postfix({
            'L':    f'{avg_loss:.3f}',
            'ELO':  f'{current_elo:.0f}',
            'ε':    f'{cube_epsilon:.2f}',
            'Cube': (
                f'D/G:{stats["doubles"]/n_g:.1f} '
                f'Tk:{stats["takes"]/n_d:.0%} '
                f'VD:{stats["sum_val_double"]/n_d:.2f}'
            )
        })

        if train_step % Config.ELO_EVAL_INTERVAL == 0:
            tqdm.write("--- Epoch Phase 3: Evaluating (Interval Reached) ---")
            model.eval()
            best_model.eval()

            total_wins, total_games, opponent_elo, wins_vs_best, n_vs_best = evaluate_combined(
                model         = model,
                best_model    = best_model,
                baseline_model= baseline_model,
                best_elo      = best_elo,
                baseline_elo  = baseline_elo,
                total_games   = Config.GATE_GAMES,
                device        = 'cpu',
                baseline_config_path = baseline_config_path if baseline_model is not None else None,
                equity_table  = equity_table,
            )

            old_elo     = current_elo
            current_elo = update_elo(current_elo, opponent_elo, total_wins, total_games)
            promoted, gate_rate = passes_gate(wins_vs_best, n_vs_best, total_wins, total_games)
            if promoted:
                old_elo = best_elo
                current_elo = promoted_elo(best_elo, wins_vs_best, n_vs_best)
                best_elo = current_elo
            tqdm.write(
                f"   -> Eval: {int(total_wins)}/{total_games} wins | "
                f"opp_elo={opponent_elo:.0f} | "
                f"ELO: {old_elo:.0f} -> {current_elo:.0f}"
            )

            tqdm.write(
                f"   -> Gate: {gate_rate:.1%} vs best "
                f"(threshold > {Config.GATE_WIN_RATE:.1%}) -> "
                f"{'PROMOTE' if promoted else 'keep best'}"
            )
            if promoted:
                load_model_state_dict(best_model, get_model_state_dict(model))
                save_checkpoint(model, optimizer, train_step, best_elo, avg_loss, best_path)
                tqdm.write(f"  --> New Best Model Saved (ELO {best_elo:.0f})")

            save_checkpoint(model, optimizer, train_step, current_elo, avg_loss, latest_path)
            equity_table.save(equity_path)
            
            # Optionally print equity table for inspection
            equity_table.print_table()
            
            model.train()

    pbar.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()