import torch
import torch.nn as nn
from model import LATENT_DIM

# ======================================================
# 1. FedTSRGNet (BiLSTM + TCN + GAN, NO regularizer)
# ======================================================
class FedTSRGNet(nn.Module):
    def __init__(self, input_dim, classes=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, 256,
            num_layers=2, batch_first=True, bidirectional=True
        )

        self.tcn = nn.Sequential(
            nn.Conv1d(512, 128, 3, padding=2, dilation=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
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


# ======================================================
# 2. FedTrust (BiLSTM + GAN, no TCN)
# ======================================================
class FedTrustNet(nn.Module):
    def __init__(self, input_dim, classes=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, 256,
            num_layers=2, batch_first=True, bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
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
        return self.classifier(x.mean(dim=1))

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)


# ======================================================
# 3. ADGAN (TCN + GAN only)
# ======================================================
class ADGAN(nn.Module):
    def __init__(self, input_dim, classes=2):
        super().__init__()

        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=2, dilation=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
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
        x = self.tcn(x.transpose(1, 2)).mean(dim=2)
        return self.classifier(x)

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)


# ======================================================
# 4. FedGAN-IDS (MLP + GAN)
# ======================================================
class FedGANIDS(nn.Module):
    def __init__(self, input_dim, classes=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
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
        return self.classifier(x.mean(dim=1))

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)
