import torch
import torch.nn as nn
import numpy as np
import os
from src.backgammon.engine import BackgammonGame
from src.backgammon.mcts import MCTS
from src.backgammon.model import get_model
from src.backgammon.config import Config
from src.backgammon.utils.train import train_batch
from src.backgammon.replay_buffer import SimpleReplayBuffer

def run_regression_suite():
    device = torch.device("cpu")
    model = get_model().to(device)
    game = BackgammonGame()
    
    print("🚀 STARTING GLOBAL REGRESSION SUITE\n" + "="*40)

    # 1. ARCHITECTURE & ACTIVATION CHECK
    print("\n[1] Architecture & Activation Check")
    has_tanh = any(isinstance(m, nn.Tanh) for m in model.modules())
    print(f"-> Value head has Tanh: {has_tanh}")
    
    # 2. REWARD SCALE SANITY
    print("\n[2] Reward Scale vs Activation")
    # Test typical points (2) vs Match Target (7)
    points = 2
    target = Config.MATCH_TARGET
    reward_mag = float(points) / (target * 2)
    
    print(f"-> Calculated Reward Magnitude: {reward_mag:.2f}")
    if has_tanh and reward_mag > 1.0:
        print("❌ CRITICAL FAILURE: Reward > 1.0 will saturate Tanh. Win rate will drop to 0%.")
    else:
        print("✅ Reward/Activation alignment looks safe.")

    # 3. ENGINE-TO-MODEL VECTORIZATION
    print("\n[3] Input Vectorization (Dtypes & Bounds)")
    board_t, ctx_t = game.get_vector(0, 0, device=device, canonical=True)
    print(f"-> Board Dtype: {board_t.dtype} (Expected: torch.int64)")
    print(f"-> Context Dtype: {ctx_t.dtype} (Expected: torch.float32)")
    
    vocab_size = model.embedding.num_embeddings
    if torch.max(board_t) >= vocab_size or torch.min(board_t) < 0:
        print(f"❌ CRITICAL FAILURE: Board values {torch.max(board_t)} exceed Embedding vocab {vocab_size}")
    else:
        print("✅ Input vectors are valid for Embedding layer.")

    # 4. CANONICAL SYMMETRY (Turn-based flipping)
    print("\n[4] Canonical Perspective Symmetry")
    # P1 with 2 checkers on point 24
    game.reset(); game.board[23] = 2; game.turn = 1
    v1, _ = game.get_vector(0, 0, canonical=True)
    
    # P2 with 2 checkers on point 1 (Point 24 from their side)
    game.reset(); game.board[0] = -2; game.turn = -1
    v2, _ = game.get_vector(0, 0, canonical=True)
    
    if torch.equal(v1, v2):
        print("✅ Symmetry: Model sees the board identically for both players.")
    else:
        print("❌ CRITICAL: Symmetry broken! Model must learn two different games.")

    # 5. MCTS SEARCH & BACKPROP
    print("\n[5] MCTS Logic & Value Sign-Flip")
    mcts = MCTS(model, num_sims=20)
    mcts.search(game, 0, 0)
    
    if not mcts.root.children:
        print("❌ FAILURE: MCTS failed to generate legal children.")
    else:
        child = mcts.root.children[0]
        # Simulate a win backprop
        val = 0.5
        mcts._backprop(child, val)
        # Root is parent, should be -val
        if mcts.root.value_sum == -val:
            print("✅ MCTS Backprop: Value correctly flips signs at root.")
        else:
            print(f"❌ MCTS Backprop: Sign flip failed. Root sum: {mcts.root.value_sum}")

    # 6. POLICY INDEXING (Bar/Off Mapping)
    print("\n[6] Policy Index Mapping")
    game.reset(); game.bar[1] = 1; game.turn = -1; game.dice = [1, 2] # P2 from bar
    legals = game.get_legal_moves()
    
    found_bar = False
    for move in legals:
        # Canonical: P2 entering from bar should look like index 24
        (src, dst), _ = game.real_action_to_canonical(move)
        s_idx = 24 if src == "bar" else src
        if s_idx == 24: found_bar = True
    
    if found_bar:
        print("✅ Policy: 'Bar' moves correctly map to canonical index 24.")
    else:
        print("❌ Policy: Mapping logic failed to identify Bar entry.")

    # 7. TRAINING LOOP (Optimization Check)
    print("\n[7] Optimization Step (One-batch Update)")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = SimpleReplayBuffer(100)
    
    # Create fake transition
    target_f, target_t = torch.zeros(26), torch.zeros(26)
    target_f[12] = 1.0; target_t[15] = 1.0
    buffer.add((board_t, ctx_t, None, 1.0, False, (target_f, target_t)))
    
    initial_loss = None
    try:
        scaler = torch.amp.GradScaler(enabled=False)
        for _ in range(5):
            loss, _ = train_batch(model, optimizer, buffer, 1, device, scaler)
            if initial_loss is None: initial_loss = loss
        
        if loss < initial_loss:
            print(f"✅ Training: Loss decreased from {initial_loss:.4f} to {loss:.4f}.")
        else:
            print("⚠️ Training: Loss did not decrease. Check optimizer/learning rate.")
    except Exception as e:
        print(f"❌ Training Loop Failed: {e}")

    print("\n" + "="*40 + "\n🏁 REGRESSION COMPLETE")

if __name__ == "__main__":
    run_regression_suite()