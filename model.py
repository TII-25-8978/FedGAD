import torch
import torch.nn as nn

LATENT_DIM = 100
LSTM_HIDDEN = 256
TCN_CHANNELS = 128
DROPOUT = 0.3

class TCNBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=2, dilation=2)
        self.bn = nn.BatchNorm1d(c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BiLSTMTCNGAN(nn.Module):
    def __init__(self, input_dim, classes=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, LSTM_HIDDEN,
            num_layers=2, batch_first=True, bidirectional=True
        )

        self.tcn = nn.Sequential(
            nn.Conv1d(2 * LSTM_HIDDEN, TCN_CHANNELS, 3, padding=2, dilation=2),
            nn.BatchNorm1d(TCN_CHANNELS),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(TCN_CHANNELS, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, classes)
        )

        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.tcn(x.transpose(1, 2)).mean(dim=2)
        return self.classifier(x)

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)