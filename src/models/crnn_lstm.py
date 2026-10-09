import torch
import torch.nn as nn

from .cnn_backbone import CNNBackbone
from .lstm_encoder import LSTMEncoder
from .attention import Attention
from .classifier import Classifier


class SharedEncoder(nn.Module):
    """
    Shared Encoder capable of processing both Mel Spectrograms and Swaragrams.
    Extracts a temporal sequence and pools it via Attention into a context vector.
    """
    def __init__(self, input_type="swaragram", lstm_hidden_size=256, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.input_type = input_type
        
        # Configure backbone based on input type
        if input_type == "mel":
            self.backbone = CNNBackbone(pool_kernel=2)
            lstm_input_size = 2048 # (256 channels * 8 freq bins)
        elif input_type == "swaragram":
            self.backbone = CNNBackbone(pool_kernel=(1, 2))
            lstm_input_size = 3072 # (256 channels * 12 freq bins)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        # Sequence Encoder
        self.encoder = LSTMEncoder(
            input_channels=lstm_input_size,
            hidden_channels=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True
        )

        # Temporal Attention
        self.attention = Attention(
            hidden_size=lstm_hidden_size * 2
        )
        
        self.output_dim = lstm_hidden_size * 2

    def forward(self, x):
        # CNN
        x = self.backbone(x)

        # (B,C,H,W) -> (B,W,C,H)
        x = x.permute(0, 3, 1, 2)

        # (B,W,C,H) -> (B,W,C*H)
        B, T, C, H = x.shape
        x = x.reshape(B, T, C * H)

        # BiLSTM
        x = self.encoder(x)

        # Attention
        context_vector = self.attention(x)
        return context_vector


class CRNN(nn.Module):
    """
    The full Dual-Task ready model combining the SharedEncoder with the MelodicOriginHead.
    """
    def __init__(
        self,
        input_type="swaragram",
        lstm_hidden_size=256,
        lstm_layers=2,
        num_classes=38,
        classifier_hidden=256,
        dropout=0.5
    ):
        super().__init__()
        
        # Shared Representation Learner
        self.shared_encoder = SharedEncoder(
            input_type=input_type,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout
        )

        # Melodic Origin Head (Predicts Top-3 candidate ragas)
        self.melodic_origin_head = Classifier(
            input_size=self.shared_encoder.output_dim,
            hidden_size=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # self.folk_genre_head = ... (Future Work)

    def forward(self, x):
        context = self.shared_encoder(x)
        return self.melodic_origin_head(context)# Assemble your CRNN architecture here
