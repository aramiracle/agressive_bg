"""Neural network models for backgammon AI."""

import torch
import torch.nn as nn
from src.backgammon.config import Config


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for sequence models."""
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions)


class BackgammonTransformer(nn.Module):
    """
    Transformer model for backgammon using Pre-LN architecture.
    
    Input:
        - board_seq: [B, 28] tokenized board state
        - context: [B, 4] game context (dice, turn, etc.)
    
    Output:
        - p_from: [B, NUM_ACTIONS] policy logits for source position
        - p_to: [B, NUM_ACTIONS] policy logits for target position
        - v: [B, 1] value estimate in [-1, 1]
        - cube: [B, 2] cube decision logits
    """
    def __init__(self, config=None):
        super().__init__()
        cfg = config if config is not None else Config
        d = cfg.D_MODEL
        
        # Input embeddings
        self.embedding = nn.Embedding(cfg.EMBED_VOCAB_SIZE, d)
        self.ctx_proj = nn.Linear(cfg.CONTEXT_SIZE, d)
        self.pos_encoder = LearnedPositionalEncoding(d, cfg.MAX_SEQ_LEN)
        
        # Transformer encoder with Pre-LN (norm_first=True)
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
        
        # Output heads
        self.policy_from = nn.Linear(d, cfg.NUM_ACTIONS)
        self.policy_to = nn.Linear(d, cfg.NUM_ACTIONS)
        
        self.value_head = nn.Sequential(
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 1),
            nn.Tanh()
        )
        
        self.cube_head = nn.Sequential(
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, board_seq, context):
        # Board embedding: [B, 28] -> [B, 28, D]
        x_board = self.embedding(board_seq)
        
        # Context projection: [B, 4] -> [B, 1, D]
        x_ctx = self.ctx_proj(context).unsqueeze(1)
        
        # Concatenate: [B, 29, D] (context token + board tokens)
        x = torch.cat([x_ctx, x_board], dim=1)
        
        # Add positional encoding and transform
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Use context token (position 0) for global predictions
        global_feat = self.out_norm(x[:, 0, :])
        
        # Compute outputs
        p_from = self.policy_from(global_feat)
        p_to = self.policy_to(global_feat)
        v = self.value_head(global_feat)
        cube = self.cube_head(global_feat)
        
        return p_from, p_to, v, cube
    
    def count_parameters(self):
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock1D(nn.Module):
    """1D Residual Block with pre-activation for CNN backbone."""
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        return out + residual


class BackgammonCNN(nn.Module):
    """
    CNN model (ResNet-style) for backgammon.
    
    Uses 1D convolutions treating board as a sequence with channel dimension.
    """
    def __init__(self, config=None):
        super().__init__()
        cfg = config if config is not None else Config
        d = cfg.D_MODEL
        
        # Input embeddings
        self.embedding = nn.Embedding(cfg.EMBED_VOCAB_SIZE, d)
        self.ctx_proj = nn.Linear(cfg.CONTEXT_SIZE, d)
        
        # Initial projection
        self.input_norm = nn.BatchNorm1d(d)
        
        # 1D ResNet Backbone
        self.blocks = nn.ModuleList([
            ResidualBlock1D(d, cfg.CNN_KERNEL, cfg.DROPOUT)
            for _ in range(cfg.CNN_BLOCKS)
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output heads
        self.policy_from = nn.Linear(d, cfg.NUM_ACTIONS)
        self.policy_to = nn.Linear(d, cfg.NUM_ACTIONS)
        
        self.value_head = nn.Sequential(
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 1),
            nn.Tanh()
        )
        
        self.cube_head = nn.Sequential(
            nn.Linear(d, cfg.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(cfg.VALUE_HIDDEN, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights appropriately for each layer type."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, board_seq, context):
        # Board embedding: [B, 28] -> [B, 28, D]
        x_board = self.embedding(board_seq)
        
        # Context projection: [B, 4] -> [B, 1, D]
        x_ctx = self.ctx_proj(context).unsqueeze(1)
        
        # Concatenate: [B, 29, D]
        x = torch.cat([x_ctx, x_board], dim=1)
        
        # Transpose for Conv1d: [B, D, 29]
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling: [B, D, 29] -> [B, D]
        x = self.global_pool(x).squeeze(-1)
        
        # Compute outputs
        p_from = self.policy_from(x)
        p_to = self.policy_to(x)
        v = self.value_head(x)
        cube = self.cube_head(x)
        
        return p_from, p_to, v, cube
    
    def count_parameters(self):
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model():
    """Factory function to create model based on Config.MODEL_TYPE."""
    if Config.MODEL_TYPE == "transformer":
        model = BackgammonTransformer()
    elif Config.MODEL_TYPE == "cnn":
        model = BackgammonCNN()
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {Config.MODEL_TYPE}")
    
    # print(f"📦 Created {Config.MODEL_TYPE.upper()} model with {model.count_parameters():,} parameters")
    return model

