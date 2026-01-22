"""ELO rating system for model evaluation."""

import torch
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
from src.backgammon.config import Config
from src.backgammon.mcts import MCTS

def calculate_expected_score(player_elo, opponent_elo):
    """Calculate expected score using standard ELO formula."""
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / Config.ELO_SCALE))

def update_elo(current_elo, opponent_elo, wins, total_games):
    """Update ELO based on match results."""
    if total_games == 0:
        return current_elo

    actual = wins / total_games
    expected = calculate_expected_score(current_elo, opponent_elo)
    delta = Config.ELO_K * (actual - expected)

    delta = max(-Config.ELO_SCALE, min(Config.ELO_SCALE, delta))
    return current_elo + delta * total_games

def play_single_game(game, mcts_a, mcts_b, a_is_white, max_moves=1000):
    """
    Play a single game between two MCTS agents.
    """
    game.reset()
    mcts_a.reset()
    mcts_b.reset()
    
    move_count = 0

    while move_count < max_moves:
        winner, _ = game.check_win()
        if winner != 0:
            if a_is_white:
                return 1.0 if winner == 1 else 0.0
            else:
                return 1.0 if winner == -1 else 0.0

        game.roll_dice()

        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                break

            a_turn = (game.turn == 1 and a_is_white) or (game.turn == -1 and not a_is_white)
            mcts = mcts_a if a_turn else mcts_b
            
            root = mcts.search(game, 0, 0)

            if root.children:
                # Fixed: Accessing object attributes .visits and .action
                best_child = max(root.children, key=lambda node: node.visits)
                action = best_child.action
            else:
                action = legal[0]

            game.step_atomic(action)
            mcts.advance_to_child(action)
            move_count += 1
            
            if game.check_win()[0] != 0:
                break

        if game.check_win()[0] == 0:
            game.switch_turn()
            mcts_a.reset()
            mcts_b.reset()

    return 0.5 # Draw/Timeout

def _worker_play_game(game_instance, model_a, model_b, device, i):
    """
    Each worker gets a reference to the shared models in RAM.
    """
    # 1. Critical for CPU performance: prevent thread oversubscription
    torch.set_num_threads(1)
    
    # 2. Local MCTS initialization
    mcts_a = MCTS(model_a, device=device)
    mcts_b = MCTS(model_b, device=device)
    
    a_is_white = (i % 2 == 0)
    
    # play_single_game handles game.reset()
    return play_single_game(game_instance, mcts_a, mcts_b, a_is_white)

def evaluate_vs_opponent(game, model_a, model_b, num_games, device='cpu', num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()

    # CRITICAL: Move models to CPU and move to SHARED MEMORY
    model_a.to('cpu').eval()
    model_b.to('cpu').eval()
    model_a.share_memory()  # No more pickling overhead!
    model_b.share_memory()

    # Use 'spawn' to ensure a clean state and avoid deadlocks
    # This is the recommended method for PyTorch multiprocessing
    ctx = mp.get_context('spawn')
    
    # Partial freezes the objects that don't change
    worker_func = partial(_worker_play_game, game, model_a, model_b, 'cpu')
    
    wins = 0.0
    with ctx.Pool(processes=num_processes) as pool:
        # imap_unordered makes the tqdm bar update smoothly
        results = list(tqdm(
            pool.imap_unordered(worker_func, range(num_games)), 
            total=num_games, 
            desc=f"Parallel ELO ({num_processes} cores)",
            dynamic_ncols=True,
            leave=False
        ))

    wins = sum(results)
    return wins, num_games