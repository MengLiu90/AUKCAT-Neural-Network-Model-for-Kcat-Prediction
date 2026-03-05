# model.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=0)
        self.bnc = nn.BatchNorm1d(16)

        self.embedding_dim = 16
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=4, dim_feedforward=32, batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)

        output_length = self.output_dim(input_size)
        self.fc_input_dim = output_length * self.embedding_dim

        self.layer1 = nn.Linear(self.fc_input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.layer2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.layer3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.output = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                       # (batch, 1, input_size)
        x = torch.relu(self.bnc(self.conv1(x)))  # (batch, 16, L)

        x = x.permute(2, 0, 1)                   # (L, batch, 16)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)  # (batch, L*16)

        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = torch.relu(self.bn3(self.layer3(x)))
        return self.output(x)                    # (batch, 1)

    def output_dim(self, input_length: int) -> int:
        k = self.conv1.kernel_size[0]
        s = self.conv1.stride[0]
        p = self.conv1.padding[0]
        return (input_length - k + 2 * p) // s + 1