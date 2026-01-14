import torch

class Config:
    # --- Board / Game Constants ---
    NUM_POINTS = 24
    CHECKERS_PER_PLAYER = 15
    HOME_SIZE = 6
    DICE_SIDES = 6
    
    BAR_IDX = 24
    OFF_IDX = 25
    NUM_ACTIONS = 26
    
    BOARD_SEQ_LEN = 28
    EMBED_VOCAB_SIZE = 31
    EMBED_OFFSET = 15
    CONTEXT_SIZE = 4
    
    INITIAL_SETUP = {
        0: -2, 5: 5, 7: 3, 11: -5,
        12: 5, 16: -3, 18: -5, 23: 2,
    }
    
    # --- Game Rules ---
    MATCH_TARGET = 15
    MAX_TURNS = 400  # Reduced from 500 to prune stalled games faster
    
    # --- Rewards ---
    R_WIN = 1.0
    R_GAMMON = 3.0
    R_BACKGAMMON = 5.0
    
    # --- Model Selection ---
    MODEL_TYPE = "transformer"
    
    # --- Model Params ---
    D_MODEL = 64
    DROPOUT = 0.1
    VALUE_HIDDEN = 32
    MAX_SEQ_LEN = BOARD_SEQ_LEN + 1
    
    # Transformer
    N_HEAD = 4
    N_LAYERS = 2  # Increased layers slightly for stability, decreased MCTS
    DIM_FEEDFORWARD = 128
    
    # CNN
    CNN_BLOCKS = 3
    CNN_KERNEL = 3
    
    # --- MCTS / Search (OPTIMIZED) ---
    NUM_SIMULATIONS = 10     # Reduced from 20 (Speed up generation 2x)
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPS = 0.25
    
    # --- ELO Rating ---
    INITIAL_ELO = 800
    ELO_K = 32
    ELO_SCALE = 400.0
    ELO_EVAL_INTERVAL = 50   # Evaluate less often to focus on training
    ELO_EVAL_GAMES = 10
    
    # --- Training (OPTIMIZED) ---
    NUM_WORKERS = 12         # Keep high
    BATCH_SIZE = 256         # Increased from 64 (Better GPU utilization)
    BUFFER_SIZE = 5000      # Larger buffer for stable training
    LR = 3e-4
    TRAIN_STEPS = 2000       # More steps, but they will happen faster
    LOSS_AVG_WINDOW = 100
    
    # Device Logic: Trainer uses CUDA, Workers use CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKER_DEVICE = "cpu"    # Force workers to CPU to avoid CUDA context switch overhead
    
    CHECKPOINT_DIR = "checkpoints"
    SAVE_INTERVAL = 100