import torch
import torch.nn as nn
import numpy as np
from src.backgammon.config import Config
from src.backgammon.engine import BackgammonGame
from src.backgammon.model import get_model
from src.backgammon.mcts import MCTS
from src.backgammon.utils.cube import get_learned_cube_decision

def run_comprehensive_diagnostic():
    device = torch.device("cpu")
    print(f"--- Initialization ---")
    try:
        model = get_model().to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    game = BackgammonGame()
    
    # =========================================================
    # 1. MODEL SIGNATURE & UNPACKING
    # =========================================================
    print("\n--- 1. Model Signature Check ---")
    board_t, ctx_t = game.get_vector(0, 0, device=device)
    # Ensure board is Long for Embedding layers
    board_input = board_t.unsqueeze(0).long() 
    ctx_input = ctx_t.unsqueeze(0).float()
    
    with torch.no_grad():
        outputs = model(board_input, ctx_input)
    
    num_outputs = len(outputs)
    print(f"Model returned {num_outputs} values.")
    
    # Based on train.py/cube.py: (p_from, p_to, value, cube_logits)
    expected_outputs = 4
    if num_outputs != expected_outputs:
        print(f"❌ WARNING: Expected {expected_outputs} outputs, got {num_outputs}.")
    else:
        p_from, p_to, val, cube = outputs
        print(f"✅ p_from shape: {p_from.shape}")
        print(f"✅ p_to shape:   {p_to.shape}")
        print(f"✅ value shape:  {val.shape}")
        print(f"✅ cube shape:   {cube.shape}")

    # =========================================================
    # 2. DATA TYPE & EMBEDDING ALIGNMENT
    # =========================================================
    print("\n--- 2. Data Type Alignment ---")
    print(f"Engine board dtype: {board_t.dtype}")
    
    # Check if embedding layer exists and its requirements
    if hasattr(model, 'embedding'):
        emb_weight = model.embedding.weight
        print(f"Embedding weight dtype: {emb_weight.dtype}")
        print(f"Embedding vocab size:   {model.embedding.num_embeddings}")
        
        max_board_val = torch.max(board_t).item()
        if max_board_val >= model.embedding.num_embeddings:
            print(f"❌ CRITICAL: Board contains value {max_board_val} >= vocab size!")
        else:
            print(f"✅ Board values are within embedding range.")

    # =========================================================
    # 3. POLICY INDEX MAPPING
    # =========================================================
    print("\n--- 3. Policy Index Mapping (Bar/Off) ---")
    # Test 'bar' -> 24 and 'off' -> 25
    test_cases = [
        # (real_src, real_dst, expected_src_idx, expected_dst_idx)
        ("bar", 5, 24, 5),
        (20, "off", 20, 25),
        (0, 1, 0, 1)
    ]
    
    for src, dst, exp_s, exp_d in test_cases:
        # Mimic the logic in play_self_play_match
        s_idx = 24 if src == "bar" else src
        e_idx = 25 if dst == "off" else dst
        if s_idx == exp_s and e_idx == exp_d:
            print(f"✅ Mapping {src}->{dst} correctly went to {s_idx}->{e_idx}")
        else:
            print(f"❌ Mapping FAILED for {src}->{dst}. Got {s_idx}->{e_idx}")

    # =========================================================
    # 4. MCTS CONSISTENCY
    # =========================================================
    print("\n--- 4. MCTS Integration Check ---")
    mcts = MCTS(model, device=device, num_sims=5)
    try:
        # This will trigger mcts.model calls
        mcts.search(game, 0, 0)
        print("✅ MCTS Search completed without unpacking errors.")
    except ValueError as ve:
        print(f"❌ MCTS Unpacking Error: {ve}")
    except Exception as e:
        print(f"❌ MCTS Generic Error: {e}")

    # =========================================================
    # 5. REWARD SIGN & MAGNITUDE
    # =========================================================
    print("\n--- 5. Reward Perspective Check ---")
    winner = 1  # P1 wins
    points = 2
    target = Config.MATCH_TARGET
    reward_mag = (float(points) + target) / target
    
    # Simulation: P1 made a move. Winner is 1.
    p1_is_winner = (1 == winner)
    p1_reward = reward_mag if p1_is_winner else -reward_mag
    
    # Simulation: P2 made a move. Winner is 1.
    p2_is_winner = (-1 == winner)
    p2_reward = reward_mag if p2_is_winner else -reward_mag

    print(f"Reward Magnitude: {reward_mag:.2f}")
    if p1_reward > 0 and p2_reward < 0:
        print("✅ Reward signs correctly perspectival (P1+, P2- for P1 win).")
    else:
        print(f"❌ Reward signs WRONG. P1: {p1_reward}, P2: {p2_reward}")
        
    if reward_mag > 1.0:
        has_tanh = any("Tanh" in str(m) for m in model.modules())
        if has_tanh:
            print("⚠️ WARNING: Value head uses Tanh but rewards > 1.0. This causes 0% win rate due to gradient saturation.")

    # =========================================================
    # 6. CUBE LOGIC
    # =========================================================
    print("\n--- 6. Cube Decision Integration ---")
    try:
        action, probs = get_learned_cube_decision(model, game, device, 0, 0, stochastic=False)
        print(f"✅ Cube decision: {action}, Probs: {probs.tolist()}")
    except Exception as e:
        print(f"❌ Cube decision FAILED: {e}")

    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    run_comprehensive_diagnostic()