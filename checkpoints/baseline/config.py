"""Frozen architecture of the nets used as the training baseline.

These weights are the stage-1 champion: a 10-layer transformer with a
5-feature context and a 6-way outcome head. HEAD_KIND tells the loader to
rebuild that net rather than the older policy-and-value architecture.
"""


class Config:
    MODEL_TYPE = "transformer"
    HEAD_KIND = "outcome"

    NUM_ACTIONS = 26
    BOARD_SEQ_LEN = 28
    EMBED_VOCAB_SIZE = 31
    CONTEXT_SIZE = 5
    NUM_OUTCOMES = 6
    D_MODEL = 128
    DROPOUT = 0.1
    VALUE_HIDDEN = 64
    MAX_SEQ_LEN = BOARD_SEQ_LEN + 1

    N_HEAD = 16
    N_LAYERS = 10
    DIM_FEEDFORWARD = 256

    CNN_BLOCKS = 4
    CNN_KERNEL = 3
