from __future__ import annotations
import torch
import torch.nn as nn


class DetectionResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class MIResNet1DDetector(nn.Module):
    def __init__(self, num_leads: int = 12, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 64, 15, 2, 7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(
            DetectionResidualBlock1D(64, 64),
            DetectionResidualBlock1D(64, 64),
        )
        self.layer2 = nn.Sequential(
            DetectionResidualBlock1D(64, 128, 2),
            DetectionResidualBlock1D(128, 128),
        )
        self.layer3 = nn.Sequential(
            DetectionResidualBlock1D(128, 256, 2),
            DetectionResidualBlock1D(256, 256),
        )
        self.layer4 = nn.Sequential(
            DetectionResidualBlock1D(256, 512, 2),
            DetectionResidualBlock1D(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x)).squeeze(1)


class LocalizationResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, 1, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class MILocalizationResNet1D(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 15, 2, 7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(
            LocalizationResidualBlock1D(32, 32),
            LocalizationResidualBlock1D(32, 32),
        )
        self.layer2 = nn.Sequential(
            LocalizationResidualBlock1D(32, 64, 2),
            LocalizationResidualBlock1D(64, 64),
        )
        self.layer3 = nn.Sequential(
            LocalizationResidualBlock1D(64, 128, 2),
            LocalizationResidualBlock1D(128, 128),
        )
        self.layer4 = nn.Sequential(
            LocalizationResidualBlock1D(128, 192, 2),
            LocalizationResidualBlock1D(192, 192),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(self.dropout(x))
