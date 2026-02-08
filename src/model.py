"""Neural network models for backgammon AI (NaN-hardened, no deprecated APIs)."""

import math
import torch
import torch.nn as nn
from src.config import Config


# ---------------------------
# Stable Learned Positional Encoding
# ---------------------------
class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for sequence models."""
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions) * self.scale


# ---------------------------
# Transformer Model
# ---------------------------
class BackgammonTransformer(nn.Module):
    """
    Transformer model for backgammon using Pre-LN architecture.
    NaN-hardened for RL + MCTS usage.
    """
    def __init__(self, config=None):
        super().__init__()
        cfg = config if config is not None else Config
        d = cfg.D_MODEL

        # Input embeddings
        self.embedding = nn.Embedding(
            cfg.EMBED_VOCAB_SIZE,
            d,
            padding_idx=0
        )

        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(cfg.CONTEXT_SIZE),
            nn.Linear(cfg.CONTEXT_SIZE, d)
        )

        self.ctx_norm = nn.LayerNorm(d)
        self.pos_encoder = LearnedPositionalEncoding(d, cfg.MAX_SEQ_LEN)

        # Transformer encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.N_HEAD,
            dim_feedforward=cfg.DIM_FEEDFORWARD,
            dropout=cfg.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.N_LAYERS,
            enable_nested_tensor=False
        )

        self.out_norm = nn.LayerNorm(d)

        # Output heads (new stable weight norm API)
        self.policy_from = nn.Linear(d, cfg.NUM_ACTIONS)
        self.policy_to = nn.Linear(d, cfg.NUM_ACTIONS)

        self.value_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 1),
            nn.Tanh()
        )

        self.cube_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 2)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights safely for deep RL training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, board_seq, context):
        # Board embedding: [B, 28] -> [B, 28, D]
        x_board = self.embedding(board_seq)

        # Context projection: [B, 4] -> [B, 1, D]
        x_ctx = self.ctx_proj(context).unsqueeze(1)
        x_ctx = self.ctx_norm(x_ctx)

        # Concatenate: [B, 29, D]
        x = torch.cat([x_ctx, x_board], dim=1)

        # Positional encoding + transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Global token
        global_feat = self.out_norm(x[:, 0, :])

        # Heads
        p_from = self.policy_from(global_feat)
        p_to = self.policy_to(global_feat)
        v = self.value_head(global_feat)
        cube = self.cube_head(global_feat)

        return p_from, p_to, v, cube

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------
# Stable Residual CNN Block
# ---------------------------
class ResidualBlock1D(nn.Module):
    """1D Residual Block with stabilized pre-activation."""
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.norm1 = nn.GroupNorm(4, channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

        self.norm2 = nn.GroupNorm(4, channels)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

        self.res_scale = 0.5

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return residual + out * self.res_scale


# ---------------------------
# CNN Model
# ---------------------------
class BackgammonCNN(nn.Module):
    """
    CNN model (ResNet-style) for backgammon.
    NaN-hardened for RL training.
    """
    def __init__(self, config=None):
        super().__init__()
        cfg = config if config is not None else Config
        d = cfg.D_MODEL

        self.embedding = nn.Embedding(
            cfg.EMBED_VOCAB_SIZE,
            d,
            padding_idx=0
        )

        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(cfg.CONTEXT_SIZE),
            nn.Linear(cfg.CONTEXT_SIZE, d)
        )

        self.input_norm = nn.GroupNorm(4, d)

        self.blocks = nn.ModuleList([
            ResidualBlock1D(d, cfg.CNN_KERNEL, cfg.DROPOUT)
            for _ in range(cfg.CNN_BLOCKS)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output heads (new weight norm API)
        self.policy_from = nn.Linear(d, cfg.NUM_ACTIONS)
        self.policy_to = nn.Linear(d, cfg.NUM_ACTIONS)


        self.value_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 1),
            nn.Tanh()
        )

        self.cube_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 2)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, board_seq, context):
        x_board = self.embedding(board_seq)
        x_ctx = self.ctx_proj(context).unsqueeze(1)

        x = torch.cat([x_ctx, x_board], dim=1)

        x = x.transpose(1, 2)
        x = self.input_norm(x)

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x).squeeze(-1)

        p_from = self.policy_from(x)
        p_to = self.policy_to(x)
        v = self.value_head(x)
        cube = self.cube_head(x)

        return p_from, p_to, v, cube

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------
# Factory
# ---------------------------
def get_model():
    """Factory function to create model based on Config.MODEL_TYPE."""
    if Config.MODEL_TYPE == "transformer":
        model = BackgammonTransformer()
    elif Config.MODEL_TYPE == "cnn":
        model = BackgammonCNN()
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {Config.MODEL_TYPE}")

    return model
