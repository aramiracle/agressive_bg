# Full Trainer vs Baseline (Crawford-safe, MCTS-safe, Engine-compatible)
"""Train new model vs older baseline version with full cubing, MCTS reuse, and ELO tracking."""

import os
import torch
import torch.optim as optim
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
from src.backgammon.utils import move_to_indices
from src.backgammon.replay_buffer import get_replay_buffer
from src.backgammon.trainer import (
    play_one_game,
    finalize_history,
    get_learned_cube_decision,
    train_batch,
)

# =========================
# BASELINE LOADER
# =========================

def load_model_with_config(config_path, model_path, device):
    """Load model with its own config without affecting global Config."""
    import importlib.util
    from src.backgammon.model import BackgammonTransformer, BackgammonCNN

    spec = importlib.util.spec_from_file_location("model_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    ModelConfig = config_module.Config

    if ModelConfig.MODEL_TYPE == "transformer":
        model = BackgammonTransformer(config=ModelConfig).to(device)
    elif ModelConfig.MODEL_TYPE == "cnn":
        model = BackgammonCNN(config=ModelConfig).to(device)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {ModelConfig.MODEL_TYPE}")

    cp = load_checkpoint(model_path, model, None, device)
    elo = cp['elo'] if cp else ModelConfig.INITIAL_ELO
    model.eval()

    return model, elo


# =========================
# VS BASELINE GAME LOOP
# =========================

def play_vs_baseline(game, current_model, baseline_model, mcts_current, device):
    """
    Play game: current vs baseline with full cubing support.
    Only saves training data for current model's decisions.
    """
    game.reset()
    game.set_match_scores(0, 0)
    current_is_p1 = random.choice([True, False])

    mcts_current.reset()
    mcts_baseline = MCTS(baseline_model, device=device)
    history = []

    for _ in range(Config.MAX_GAME_MOVES):
        winner, _ = game.check_win()
        if winner != 0: break

        is_current_turn = (game.turn == 1) == current_is_p1
        active_model = current_model if is_current_turn else baseline_model
        active_mcts = mcts_current if is_current_turn else mcts_baseline
        
        my_score = game.match_scores[game.turn]
        opp_score = game.match_scores[-game.turn]

        # =========================
        # LEARNED CUBING PHASE
        # =========================
        if game.can_double() and not game.crawford_active:
            double_choice, d_probs = get_learned_cube_decision(
                active_model, game, device, my_score, opp_score, stochastic=True
            )

            if is_current_turn:
                board_t, ctx_t = game.get_vector(my_score, opp_score, device='cpu', canonical=True)
                # Matches trainer.py schema: (board, ctx, action, turn, is_cube, probs)
                history.append((board_t, ctx_t, None, game.turn, True, d_probs))

            if double_choice == 1:
                game.switch_turn() 
                is_responder_current = (game.turn == 1) == current_is_p1
                resp_model = current_model if is_responder_current else baseline_model
                
                take_choice, t_probs = get_learned_cube_decision(
                    resp_model, game, device, opp_score, my_score, stochastic=True
                )

                if is_responder_current:
                    board_t, ctx_t = game.get_vector(opp_score, my_score, device='cpu', canonical=True)
                    history.append((board_t, ctx_t, None, game.turn, True, t_probs))

                game.switch_turn() 

                if take_choice == 1:
                    game.apply_double()
                else:
                    winner, points = game.handle_cube_refusal()
                    current_won = (winner == 1) == current_is_p1
                    # Use imported finalize_history for reward consistency
                    return finalize_history(history, current_won, points), current_won

        # =========================
        # MOVEMENT PHASE
        # =========================
        game.roll_dice()
        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                game.dice = []
                break

            root = active_mcts.search(game, 0, 0)
            if root.children:
                visits = torch.tensor([c.visits for c in root.children], dtype=torch.float)
                probs = visits / visits.sum() if visits.sum() > 0 else torch.ones(len(visits))/len(visits)
                chosen_action = root.children[torch.multinomial(probs, 1).item()].action
            else:
                chosen_action = random.choice(legal)

            if is_current_turn:
                board_t, ctx_t = game.get_vector(game.match_scores[1], game.match_scores[-1], device='cpu', canonical=True)
                target_f, target_t = torch.zeros(26), torch.zeros(26)

                if root.children:
                    for i, child in enumerate(root.children):
                        canon_act = game.real_action_to_canonical(child.action)
                        s, e = move_to_indices(canon_act[0], canon_act[1])
                        if 0 <= s < 26: target_f[s] += probs[i]
                        if 0 <= e < 26: target_t[e] += probs[i]
                
                history.append((board_t, ctx_t, None, game.turn, False, (target_f, target_t)))

            game.step_atomic(chosen_action)
            active_mcts.advance_to_child(chosen_action)
            if game.check_win()[0] != 0: break

        if game.check_win()[0] == 0:
            game.switch_turn()
            mcts_current.reset()
            mcts_baseline.reset()

    # =========================
    # FINALIZATION
    # =========================
    winner, mult = game.check_win()
    total_points = mult * game.cube
    current_won = (winner == 1) == current_is_p1

    # centralized finalize_history handles the reward sign and match normalization
    return finalize_history(history, current_won, total_points), current_won


# =========================
# TRAIN LOOP
# =========================

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

    # Load Baseline
    baseline_path = os.path.join(os.path.dirname(checkpoint_dir), Config.BASELINE_DIR, Config.BASELINE_MODEL_NAME)
    baseline_config_path = os.path.join(os.path.dirname(checkpoint_dir), Config.BASELINE_DIR, 'config.py')

    baseline_model, baseline_elo = None, Config.INITIAL_ELO
    use_baseline = False

    if os.path.exists(baseline_path) and os.path.exists(baseline_config_path):
        baseline_model, baseline_elo = load_model_with_config(baseline_config_path, baseline_path, device)
        print(f"✅ Baseline loaded: {baseline_path} (ELO: {baseline_elo:.0f})")
        use_baseline = True

    replay_buffer = get_replay_buffer(int(Config.BUFFER_SIZE * 0.6), prioritized=True)
    game = BackgammonGame()

    pbar = tqdm(total=Config.TRAIN_STEPS, initial=train_step, desc="Overall Training")

    while train_step < Config.TRAIN_STEPS:
        # COLLECT
        model.eval()
        mcts = MCTS(model, device=device)
        baseline_ratio = Config.BASELINE_SELF_PLAY_RATIO

        phase = "vs_baseline" if (use_baseline and current_elo < baseline_elo) else "self_play"
        num_self = int(Config.GAMES_PER_ITERATION * (baseline_ratio if phase == "vs_baseline" else 1.0))
        num_baseline = Config.GAMES_PER_ITERATION - num_self

        for _ in range(num_self):
            data, _ = play_one_game(game, mcts, model, device)
            replay_buffer.extend(data)

        if use_baseline and num_baseline > 0:
            for _ in range(num_baseline):
                data, _ = play_vs_baseline(game, model, baseline_model, mcts, device)
                replay_buffer.extend(data)

        if len(replay_buffer) < Config.BATCH_SIZE: continue

        # TRAIN
        model.train()
        for _ in range(Config.STEPS_PER_ITERATION):
            # This calls the patched train_batch in trainer.py with NaN protection and Smoothing
            loss, gnorm = train_batch(model, optimizer, replay_buffer, Config.BATCH_SIZE, device, scaler)
            
            train_step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss:.1f}', 'elo': f'{current_elo:.0f}', 'gnorm': f'{gnorm:.3f}'})

            # EVALUATE
            if train_step % Config.ELO_EVAL_INTERVAL == 0:
                model.eval()
                wins, total = evaluate_vs_opponent(game, model, best_model, Config.ELO_EVAL_GAMES, device)
                
                # Dynamic opponent ELO for calculation
                opp_elo = (best_elo * 0.5 + baseline_elo * 0.5) if (phase == "vs_baseline") else best_elo
                current_elo = update_elo(current_elo, opp_elo, wins, total)

                if current_elo > best_elo:
                    best_elo = current_elo
                    load_model_state_dict(best_model, get_model_state_dict(model))
                    save_checkpoint(model, optimizer, train_step, best_elo, loss, best_path)
                
                save_checkpoint(model, optimizer, train_step, current_elo, loss, latest_path)
                model.train()

    pbar.close()

if __name__ == "__main__":
    train()