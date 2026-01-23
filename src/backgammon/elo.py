"""ELO rating system for model evaluation with Learned Cubing."""

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

def get_cube_action(model, game, device, my_score=0, opp_score=0):
    """Consult the model's learned cube policy."""
    # Canonical=True ensures the model sees the board from its own perspective
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)
    
    with torch.no_grad():
        # Policy head outputs: [Move_Probs, Value, Cube_Logits]
        _, _, _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        
        # In evaluation, we take the most confident action (Argmax)
        # Index 0: No Double / Drop
        # Index 1: Double / Take
        action = torch.argmax(cube_logits.squeeze(0)).item()
    return action

def play_single_game(game, model_a, model_b, mcts_a, mcts_b, a_is_white, device, 
                     score_a, score_b, max_moves=1000):
    """Plays a single game within a match context."""
    game.reset()
    mcts_a.reset()
    mcts_b.reset()
    
    move_count = 0
    while move_count < max_moves:
        winner, points = game.check_win()
        if winner != 0: break

        # Determine perspective
        is_a_turn = (game.turn == 1 and a_is_white) or (game.turn == -1 and not a_is_white)
        p1_score, p2_score = (score_a, score_b) if is_a_turn else (score_b, score_a)
        
        active_model = model_a if is_a_turn else model_b
        opp_model = model_b if is_a_turn else model_a
        active_mcts = mcts_a if is_a_turn else mcts_b

        # ---------------- 1. Learned Doubling ----------------
        if game.can_double():
            if get_cube_action(active_model, game, device, p1_score, p2_score) == 1:
                game.switch_turn() # Opponent's turn to decide Take/Drop
                take_decision = get_cube_action(opp_model, game, device, p2_score, p1_score)
                game.switch_turn() 

                if take_decision == 1:
                    game.apply_double()
                else:
                    # Opponent drops: winner gets current cube value
                    win_side, cube_val = game.handle_cube_refusal()
                    return win_side, cube_val

        # ---------------- 2. Movement ----------------
        game.roll_dice()
        while game.dice:
            legal = game.get_legal_moves()
            if not legal: break
            
            # Pass scores to MCTS so search values reflect match equity
            root = active_mcts.search(game, p1_score, p2_score) 
            action = max(root.children, key=lambda n: n.visits).action
            
            game.step_atomic(action)
            active_mcts.advance_to_child(action)
            move_count += 1
            if game.check_win()[0] != 0: break

        if game.check_win()[0] == 0:
            game.switch_turn()

    winner, points = game.check_win()
    return winner, points * game.cube

def _worker_play_match(game_instance, model_a, model_b, device, i):
    torch.set_num_threads(1)
    mcts_a = MCTS(model_a, device=device)
    mcts_b = MCTS(model_b, device=device)
    
    score_a, score_b = 0, 0
    target = Config.MATCH_TARGET # e.g., 7 or 11
    a_is_white = (i % 2 == 0)

    while score_a < target and score_b < target:
        winner, points = play_single_game(
            game_instance, model_a, model_b, mcts_a, mcts_b, 
            a_is_white, device, score_a, score_b
        )
        
        if (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
            score_a += points
        else:
            score_b += points
            
    return 1.0 if score_a >= target else 0.0

def evaluate_vs_opponent(game, model_a, model_b, num_games, device='cpu', num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Move models to device and share memory for multiprocessing efficiency
    model_a.to(device).eval()
    model_b.to(device).eval()
    model_a.share_memory()
    model_b.share_memory()

    ctx = mp.get_context('spawn')
    worker_func = partial(_worker_play_match, game, model_a, model_b, device)
    
    wins = 0.0
    # Create the tqdm object explicitly to update it inside the loop
    pbar = tqdm(
        total=num_games, 
        desc=f"Parallel ELO ({num_processes} cores)",
        dynamic_ncols=True,
        leave=False
    )

    with ctx.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(worker_func, range(num_games)):
            wins += result
            pbar.update(1)
            # This updates the right-hand side of the bar with the win ratio
            pbar.set_postfix({"wins": f"{int(wins)}/{pbar.n}"})

    pbar.close()
    return wins, num_games