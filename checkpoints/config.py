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
    LABEL_SMOOTHING = 0.02
    
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
