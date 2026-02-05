import torch

class Config:
    # Board / Game
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

    # Game
    MATCH_TARGET = 7

    # Rewards
    R_WIN = 1.0
    R_GAMMON = 3.0
    R_BACKGAMMON = 5.0

    # Model
    MODEL_TYPE = "transformer"  # "transformer" or "cnn"
    D_MODEL = 128
    DROPOUT = 0.1
    VALUE_HIDDEN = 64
    MAX_SEQ_LEN = BOARD_SEQ_LEN + 1
    
    # Transformer specific
    N_HEAD = 8
    N_LAYERS = 5
    DIM_FEEDFORWARD = 256
    
    # CNN specific
    CNN_BLOCKS = 4
    CNN_KERNEL = 3

    # MCTS
    NUM_SIMULATIONS = 128
    MCTS_BATCH = 8
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPS = 0.25
    MIN_PRIOR = 1e-5

    # ========================================
    # CUBE LEARNING FIXES
    # ========================================
    
    # Cube Exploration: Force random cube decisions during training
    # This is CRITICAL to break the "never double" cycle
    CUBE_EPSILON_START = 0.25  # Start with 20% random cube decisions
    CUBE_EPSILON_END = 0.02    # Decay to 5% random decisions
    CUBE_EPSILON_DECAY_STEPS = 400000  # Linear decay over 200k steps
    
    # Cube Loss Weight: Emphasize learning cube decisions
    # Higher weight = model pays more attention to cube errors
    CUBE_LOSS_WEIGHT = 2.0  # 2x weight on cube loss vs movement loss
    
    # Curriculum Learning Stages (Optional - can enable later)
    # Progressive difficulty in cube learning
    CUBE_CURRICULUM_ENABLED = False
    CUBE_CURRICULUM_STAGES = [
        {'steps': 0,      'epsilon': 0.3, 'cube_weight': 3.0},  # High exploration early
        {'steps': 50000,  'epsilon': 0.2, 'cube_weight': 2.5},
        {'steps': 100000, 'epsilon': 0.15, 'cube_weight': 2.0},
        {'steps': 200000, 'epsilon': 0.1, 'cube_weight': 1.5},
        {'steps': 300000, 'epsilon': 0.05, 'cube_weight': 1.0},  # Low exploration late
    ]

    # ELO
    INITIAL_ELO = 0
    ELO_K = 4
    ELO_SCALE = 400.0
    ELO_EVAL_INTERVAL = 1000
    ELO_EVAL_GAMES = 50

    # Training
    # OPTIMIZATION: Increased data generation, reduced overfitting loop
    MATCHES_PER_ITERATION = 4
    TRAIN_UPDATES_PER_ITER = 100
    
    BATCH_SIZE = 512 if torch.cuda.is_available() else 128
    BUFFER_SIZE = 100000
    KL_EPSILON = 1e-6
    LABEL_SMOOTHING = 0.02  # For movement policy only, NOT for cube
    
    # OPTIMIZATION: Increased LR for faster convergence
    LR = 1e-5
    GRAD_CLIP = 1.0
    WEIGHT_DECAY = 1e-4
    TRAIN_STEPS = 1000000
    LOSS_AVG_WINDOW = 100
    MAX_GAME_MOVES = 2000

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoints
    CHECKPOINT_DIR = "checkpoints"
    BASELINE_DIR = "checkpoints_v1"
    BASELINE_MODEL_NAME = "best_model.pt"
    BASELINE_SWITCH_ON_SURPASS = True
    BASELINE_SELF_PLAY_RATIO = 0.5
    
    # ========================================
    # CUBE DIAGNOSTICS (Optional Logging)
    # ========================================
    CUBE_LOGGING_ENABLED = True  # Log cube action distributions
    CUBE_LOG_INTERVAL = 100      # Log every N training steps