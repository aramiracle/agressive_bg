"""
Neural network models for backgammon AI.
"""

import torch
import torch.nn as nn
from config import Config


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions)


class BackgammonTransformer(nn.Module):
    """Transformer model for backgammon."""
    def __init__(self):
        super().__init__()
        d = Config.D_MODEL
        
        self.embedding = nn.Embedding(Config.EMBED_VOCAB_SIZE, d)
        self.ctx_proj = nn.Linear(Config.CONTEXT_SIZE, d)
        self.pos_encoder = LearnedPositionalEncoding(d, Config.MAX_SEQ_LEN)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=Config.N_HEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, Config.N_LAYERS)
        self.out_norm = nn.LayerNorm(d)
        
        # Heads
        self.policy_from = nn.Linear(d, Config.NUM_ACTIONS)
        self.policy_to = nn.Linear(d, Config.NUM_ACTIONS)
        self.value_head = nn.Sequential(
            nn.Linear(d, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 1),
            nn.Tanh()
        )
        self.cube_head = nn.Sequential(
            nn.Linear(d, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 2)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, board_seq, context):
        # [B, 28] -> [B, 28, D]
        x_board = self.embedding(board_seq)
        # [B, 4] -> [B, 1, D]
        x_ctx = self.ctx_proj(context).unsqueeze(1)
        # [B, 29, D]
        x = torch.cat([x_ctx, x_board], dim=1)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Use context token for global prediction
        global_feat = self.out_norm(x[:, 0, :])
        
        p_from = self.policy_from(global_feat)
        p_to = self.policy_to(global_feat)
        v = self.value_head(global_feat)
        cube = self.cube_head(global_feat)
        return p_from, p_to, v, cube
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock1D(nn.Module):
    """1D Residual Block for CNN."""
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.act(out)


class BackgammonCNN(nn.Module):
    """
    CNN model (ResNet-style) for backgammon.
    Treats the board sequence as a 1D signal with channels equal to embedding dim.
    """
    def __init__(self):
        super().__init__()
        d = Config.D_MODEL
        
        self.embedding = nn.Embedding(Config.EMBED_VOCAB_SIZE, d)
        self.ctx_proj = nn.Linear(Config.CONTEXT_SIZE, d)
        
        # 1D ResNet Backbone
        self.blocks = nn.ModuleList([
            ResidualBlock1D(d, Config.CNN_KERNEL, Config.DROPOUT)
            for _ in range(Config.CNN_BLOCKS)
        ])
        
        # Heads
        # Flatten input: D_MODEL * MAX_SEQ_LEN (29)
        flat_size = d * Config.MAX_SEQ_LEN
        
        self.policy_from = nn.Linear(flat_size, Config.NUM_ACTIONS)
        self.policy_to = nn.Linear(flat_size, Config.NUM_ACTIONS)
        
        self.value_head = nn.Sequential(
            nn.Linear(flat_size, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 1),
            nn.Tanh()
        )
        
        self.cube_head = nn.Sequential(
            nn.Linear(flat_size, Config.VALUE_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.VALUE_HIDDEN, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, board_seq, context):
        # Embed Board: [B, 28] -> [B, 28, D]
        x_board = self.embedding(board_seq)
        
        # Embed Context: [B, 4] -> [B, D] -> [B, 1, D]
        x_ctx = self.ctx_proj(context).unsqueeze(1)
        
        # Concat: [B, 29, D]
        x = torch.cat([x_ctx, x_board], dim=1)
        
        # Permute for Conv1d: [B, D, 29]
        x = x.transpose(1, 2)
        
        # Pass through ResNet blocks
        for block in self.blocks:
            x = block(x)
            
        # Flatten: [B, D*29]
        x = x.flatten(1)
        
        p_from = self.policy_from(x)
        p_to = self.policy_to(x)
        v = self.value_head(x)
        cube = self.cube_head(x)
        
        return p_from, p_to, v, cube
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model():
    """Factory function to get model based on Config."""
    if Config.MODEL_TYPE == "transformer":
        return BackgammonTransformer()
    elif Config.MODEL_TYPE == "cnn":
        return BackgammonCNN()
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {Config.MODEL_TYPE}")