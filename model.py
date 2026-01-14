"""
Lightweight neural network models for backgammon AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding - more efficient than sinusoidal for small sequences."""
    
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions)


class BackgammonTransformer(nn.Module):
    """
    Lightweight Transformer model for backgammon.
    
    Architecture:
    - Embedding layer for board positions
    - Context projection
    - Shallow transformer encoder (2 layers default)
    - Policy heads (from, to)
    - Value head
    - Cube head
    
    Parameters from Config:
    - D_MODEL: 64 (embedding dimension)
    - N_HEAD: 2 (attention heads)
    - N_LAYERS: 2 (transformer layers)
    - DIM_FEEDFORWARD: 128 (feedforward hidden dim)
    """
    
    def __init__(self):
        super().__init__()
        
        d = Config.D_MODEL
        
        # Input Embedding
        self.embedding = nn.Embedding(Config.EMBED_VOCAB_SIZE, d)
        
        # Context projection (4 values -> d_model)
        self.ctx_proj = nn.Linear(Config.CONTEXT_SIZE, d)
        
        # Learned positional encoding (more efficient for small sequences)
        self.pos_encoder = LearnedPositionalEncoding(d, Config.MAX_SEQ_LEN)
        
        # Lightweight Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=Config.N_HEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            activation='gelu',  # GELU often works better than ReLU
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, Config.N_LAYERS)
        
        # Output layer norm
        self.out_norm = nn.LayerNorm(d)
        
        # Policy Heads (lightweight)
        self.policy_from = nn.Linear(d, Config.NUM_ACTIONS)
        self.policy_to = nn.Linear(d, Config.NUM_ACTIONS)
        
        # Value Head (compact)
        self.value_head = nn.Sequential(
            nn.Linear(d, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 1),
            nn.Tanh()
        )
        
        # Cube Head (compact)
        self.cube_head = nn.Sequential(
            nn.Linear(d, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, board_seq, context):
        """
        Forward pass.
        
        Args:
            board_seq: [B, 28] int - board state indices
            context: [B, 4] float - [turn, cube, my_score, opp_score]
            
        Returns:
            p_from: [B, 26] - policy logits for source position
            p_to: [B, 26] - policy logits for target position
            v: [B, 1] - value estimate (-1 to 1)
            cube: [B, 2] - cube decision logits [no_double, double]
        """
        # Embed board positions
        x_board = self.embedding(board_seq)  # [B, 28, D]
        
        # Project context and add as first token
        x_ctx = self.ctx_proj(context).unsqueeze(1)  # [B, 1, D]
        
        # Concatenate: [context_token, board_tokens]
        x = torch.cat([x_ctx, x_board], dim=1)  # [B, 29, D]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Extract context token for predictions
        global_feat = self.out_norm(x[:, 0, :])
        
        # Heads
        p_from = self.policy_from(global_feat)
        p_to = self.policy_to(global_feat)
        v = self.value_head(global_feat)
        cube = self.cube_head(global_feat)
        
        return p_from, p_to, v, cube
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BackgammonMLP(nn.Module):
    """
    Ultra-lightweight MLP model for backgammon.
    Even faster than transformer, good for quick experiments.
    """
    
    def __init__(self):
        super().__init__()
        
        # Flatten input: 28 board positions + 4 context = 32 values
        # Each board position is embedded
        input_dim = Config.BOARD_SEQ_LEN * Config.D_MODEL + Config.CONTEXT_SIZE
        hidden = Config.D_MODEL * 2  # 128
        
        self.embedding = nn.Embedding(Config.EMBED_VOCAB_SIZE, Config.D_MODEL)
        
        # Simple MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        
        # Heads
        self.policy_from = nn.Linear(hidden, Config.NUM_ACTIONS)
        self.policy_to = nn.Linear(hidden, Config.NUM_ACTIONS)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 1),
            nn.Tanh()
        )
        
        self.cube_head = nn.Sequential(
            nn.Linear(hidden, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 2)
        )
    
    def forward(self, board_seq, context):
        # Embed and flatten board
        x_board = self.embedding(board_seq)  # [B, 28, D]
        x_board = x_board.flatten(1)  # [B, 28*D]
        
        # Concatenate with context
        x = torch.cat([x_board, context], dim=1)  # [B, 28*D + 4]
        
        # Backbone
        feat = self.backbone(x)
        
        # Heads
        p_from = self.policy_from(feat)
        p_to = self.policy_to(feat)
        v = self.value_head(feat)
        cube = self.cube_head(feat)
        
        return p_from, p_to, v, cube
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)