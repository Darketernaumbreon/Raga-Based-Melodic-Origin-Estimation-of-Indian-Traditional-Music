"""
dual_branch_crnn.py

Late-Fusion Dual-Branch CRNN for Experiment 4.

Architecture:
  Mel Branch  (CNN -> BiLSTM -> Attention) -> Mel Embedding
  Swara Branch(CNN -> BiLSTM -> Attention) -> Swara Embedding
                                          -> Concatenate
                                          -> Linear Projection
                                          -> Melodic Origin Head
"""
import torch
import torch.nn as nn

from .cnn_backbone import CNNBackbone
from .lstm_encoder import LSTMEncoder
from .attention import Attention
from .classifier import Classifier


class SingleBranch(nn.Module):
    """
    One branch of the dual-branch model.
    Processes a single input modality (Mel or Swaragram) through a full
    CNN -> BiLSTM -> Attention pipeline, producing a fixed-size embedding.
    """
    def __init__(self, input_type="mel", lstm_hidden=256, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.input_type = input_type

        if input_type == "mel":
            self.backbone = CNNBackbone(pool_kernel=2)
            lstm_input_size = 2048   # 256 ch x 8 freq bins
        elif input_type == "swaragram":
            self.backbone = CNNBackbone(pool_kernel=(1, 2))
            lstm_input_size = 3072   # 256 ch x 12 freq bins
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.encoder = LSTMEncoder(
            input_channels=lstm_input_size,
            hidden_channels=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = Attention(hidden_size=lstm_hidden * 2)
        self.output_dim = lstm_hidden * 2

    def forward(self, x):
        x = self.backbone(x)                    # (B, C, H, W)
        x = x.permute(0, 3, 1, 2)              # (B, W, C, H)
        B, T, C, H = x.shape
        x = x.reshape(B, T, C * H)             # (B, T, C*H)
        x = self.encoder(x)                     # (B, T, 2*hidden)
        x = self.attention(x)                   # (B, 2*hidden)
        return x


class DualBranchCRNN(nn.Module):
    """
    Late-Fusion Dual-Branch CRNN.

    Each modality (Mel, Swaragram) is processed independently through its own
    CNN + BiLSTM + Attention stack. The resulting embeddings are concatenated
    and projected before the Melodic Origin Head.

    Late fusion is chosen over early fusion because Mel and Swaragram capture
    structurally different information (timbral vs. melodic), and each branch
    should be free to develop its own temporal representation before they are
    combined.
    """
    def __init__(
        self,
        lstm_hidden=256,
        lstm_layers=2,
        num_classes=38,
        classifier_hidden=256,
        dropout=0.5,
        fusion_dim=512
    ):
        super().__init__()

        # Independent branches — each is a full SharedEncoder equivalent
        self.mel_branch = SingleBranch("mel", lstm_hidden, lstm_layers, dropout)
        self.swara_branch = SingleBranch("swaragram", lstm_hidden, lstm_layers, dropout)

        combined_dim = self.mel_branch.output_dim + self.swara_branch.output_dim  # 1024

        # Late-fusion projection — maps concatenated embeddings to a shared space
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Melodic Origin Head
        self.melodic_origin_head = Classifier(
            input_size=fusion_dim,
            hidden_size=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout
        )

        # self.folk_genre_head = ... (Future Work)

    def forward(self, mel, swara):
        mel_emb = self.mel_branch(mel)          # (B, D_mel)
        swara_emb = self.swara_branch(swara)    # (B, D_swara)

        combined = torch.cat([mel_emb, swara_emb], dim=1)  # (B, D_mel + D_swara)
        fused = self.fusion(combined)                        # (B, fusion_dim)
        return self.melodic_origin_head(fused)
