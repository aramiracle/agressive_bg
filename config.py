import torch

class Config:
    # --- Board / Game Constants ---
    NUM_POINTS = 24              # Board positions (0-23)
    CHECKERS_PER_PLAYER = 15     # Each player has 15 checkers
    HOME_SIZE = 6                # Home board size (6 points)
    DICE_SIDES = 6               # Standard dice (1-6)
    
    # Action space indices
    BAR_IDX = 24                 # Index for 'bar' in action encoding
    OFF_IDX = 25                 # Index for 'off' in action encoding
    NUM_ACTIONS = 26             # Total action positions (24 board + bar + off)
    
    # State encoding
    BOARD_SEQ_LEN = 28           # 24 points + 2 bar + 2 off
    EMBED_VOCAB_SIZE = 31        # Values -15 to +15 shifted to 0-30
    EMBED_OFFSET = 15            # Offset for embedding (value + 15)
    CONTEXT_SIZE = 4             # [Turn, Cube, MyScore, OppScore]
    
    # Initial board setup: (position, count) for each player
    # Positive = P1, Negative = P-1
    INITIAL_SETUP = {
        0: -2,   # P-1: 2 checkers
        5: 5,    # P1: 5 checkers
        7: 3,    # P1: 3 checkers
        11: -5,  # P-1: 5 checkers
        12: 5,   # P1: 5 checkers
        16: -3,  # P-1: 3 checkers
        18: -5,  # P-1: 5 checkers
        23: 2,   # P1: 2 checkers
    }
    
    # --- Game Rules ---
    MATCH_TARGET = 15
    MAX_TURNS = 500              # Prevent infinite loops in random play
    
    # --- Rewards (Aggressive) ---
    R_WIN = 1.0
    R_GAMMON = 3.0
    R_BACKGAMMON = 5.0
    
    # --- Model Architecture (Lightweight) ---
    D_MODEL = 32                 # Embedding dimension (was 128)
    N_HEAD = 2                   # Attention heads (was 4)
    N_LAYERS = 2                 # Transformer layers (was 4)
    DIM_FEEDFORWARD = 64        # Feedforward dimension (default: 4*D_MODEL=256, reduced)
    DROPOUT = 0.1
    VALUE_HIDDEN = 16            # Hidden size for value/cube heads (was 64)
    MAX_SEQ_LEN = BOARD_SEQ_LEN + 1  # Board + context token
    
    # --- MCTS / Search ---
    NUM_SIMULATIONS = 30         # Lower for Python speed, increase for production
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPS = 0.25
    
    # --- ELO Rating ---
    INITIAL_ELO = 800            # Starting ELO (random play baseline)
    ELO_K = 32                   # K-factor for ELO updates
    ELO_SCALE = 400.0            # ELO formula scale factor
    ELO_EVAL_INTERVAL = 25      # Evaluate ELO every N steps
    ELO_EVAL_GAMES = 10          # Games per ELO evaluation
    
    # --- Training ---
    NUM_WORKERS = 5              # Adjust based on CPU cores
    BATCH_SIZE = 64
    BUFFER_SIZE = 5000
    LR = 3e-4
    TRAIN_STEPS = 10000
    LOSS_AVG_WINDOW = 100        # Window for running average loss
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Checkpoints ---
    CHECKPOINT_DIR = "checkpoints"
    SAVE_INTERVAL = 100          # Save latest model every N steps