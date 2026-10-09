"""
Convolutional Recurrent Neural Network (CRNN) Architecture for Raag Classification.

CRNNs combine:
1. CNN layers: extract local pitch/spectral features across frequency and short time frames.
2. RNN layers (GRU or LSTM): model long-term temporal progression and melodic phrases across time.
"""

import torch
import torch.nn as nn


class RaagCRNN(nn.Module):
    """
    Scaffold for building a Convolutional Recurrent Neural Network (CRNN).

    Input shape: (Batch_Size, Channels=1, Frequency_Bins, Time_Steps)
    Output shape: (Batch_Size, num_classes)
    """
    def __init__(self, num_classes: int = 38, in_channels: int = 1, hidden_size: int = 128):
        super().__init__()
        self.num_classes = num_classes

        # TODO 1: Define 2D CNN layers to reduce frequency dimension while preserving time
        # Example: 2 or 3 Conv2d + BatchNorm + ReLU + MaxPool layers

        # TODO 2: Define Recurrent Layer (e.g., nn.GRU or nn.LSTM)
        # Note: PyTorch RNNs expect input shape (Batch, Time, Features) when batch_first=True.
        # You will need to reshape/permute the CNN output before passing into GRU/LSTM.

        # TODO 3: Define final Linear classifier over the last hidden state of the RNN
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        # TODO: Implement forward pass (CNN -> Permute -> GRU/LSTM -> Linear)
        raise NotImplementedError("Implement forward() in RaagCRNN")
