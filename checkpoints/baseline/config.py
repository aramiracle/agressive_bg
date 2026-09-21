"""Frozen architecture of the policy+value nets used as the training baseline.

Those checkpoints predate the 6-way outcome head and the 5-feature context
vector. HEAD_KIND tells the loader to rebuild that older net and adapt it to
the current search interface.
"""


class Config:
    MODEL_TYPE = "transformer"
    HEAD_KIND = "value_policy"

    NUM_ACTIONS = 26
    BOARD_SEQ_LEN = 28
    EMBED_VOCAB_SIZE = 31
    CONTEXT_SIZE = 4
    D_MODEL = 128
    DROPOUT = 0.1
    VALUE_HIDDEN = 64
    MAX_SEQ_LEN = BOARD_SEQ_LEN + 1

    N_HEAD = 16
    N_LAYERS = 12
    DIM_FEEDFORWARD = 512

    CNN_BLOCKS = 4
    CNN_KERNEL = 3
