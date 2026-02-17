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
    NUM_SIMULATIONS = 32
    MCTS_BATCH = 4
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPS = 0.25
    MIN_PRIOR = 1e-5

    # ========================================
    # CUBE LEARNING - CURRICULUM
    # ========================================
    # epsilon controls stochastic exploration of cube decisions.
    # cube_weight scales the cube head loss relative to value + policy loss.
    #
    # With the circular-gradient bug fixed, the cube head now receives a genuine
    # one-hot JS signal. We can reduce epsilon more gradually — the network will
    # actually learn from the signal rather than drift. Start higher to explore
    # the full range of cube positions (take AND drop) before committing.
    CUBE_CURRICULUM_ENABLED = True
    CUBE_CURRICULUM_STAGES = [
        {'steps': 0,      'epsilon': 0.5,  'cube_weight': 2.0},
        {'steps': 25000,  'epsilon': 0.40, 'cube_weight': 1.6},
        {'steps': 50000,  'epsilon': 0.30, 'cube_weight': 1.4},
        {'steps': 75000,  'epsilon': 0.20, 'cube_weight': 1.3},
        {'steps': 100000, 'epsilon': 0.10, 'cube_weight': 1.2},
        {'steps': 150000, 'epsilon': 0.05, 'cube_weight': 1.1},
        {'steps': 200000, 'epsilon': 0.02, 'cube_weight': 1.0},
    ]

    # CUBE_LOSS_WEIGHT: previously amplified a near-zero (circular) gradient to no
    # effect. Now that the cube head loss is a genuine JS(model || one-hot) signal,
    # 1.0 gives equal footing with policy loss. Do NOT set above ~2.0 or the cube
    # head will overwhelm the value head's gradients through shared transformer weights.
    CUBE_LOSS_WEIGHT = 1.0

    # ELO
    INITIAL_ELO = 0
    ELO_K = 4
    ELO_SCALE = 400.0
    ELO_EVAL_INTERVAL = 1000
    ELO_EVAL_GAMES = 80

    # Training
    MATCHES_PER_ITERATION = 4
    TRAIN_UPDATES_PER_ITER = 50

    BATCH_SIZE = 512 if torch.cuda.is_available() else 256
    BUFFER_SIZE = 100000
    KL_EPSILON = 1e-6
    LABEL_SMOOTHING = 0.02

    LR = 1e-5
    GRAD_CLIP = 1.0
    WEIGHT_DECAY = 1e-4
    TRAIN_STEPS = 1000000
    LOSS_AVG_WINDOW = 100
    MAX_GAME_MOVES = 2000

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SELF_PLAY_DEVICE = "cpu"  # Often faster to run env on CPU if GPU is busy training

    # Checkpoints
    CHECKPOINT_DIR = "checkpoints"
    BASELINE_DIR = "checkpoints_v1"
    BASELINE_MODEL_NAME = "best_model.pt"
    BASELINE_SWITCH_ON_SURPASS = True
    BASELINE_SELF_PLAY_RATIO = 0.5