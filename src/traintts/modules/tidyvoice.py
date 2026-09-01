import torch
from torch import Tensor, nn


class SimAMBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.relu(self.bn1(self.conv1(values)))
        hidden = self.bn2(self.conv2(hidden))
        elements = hidden.shape[2] * hidden.shape[3] - 1
        squared = (hidden - hidden.mean(dim=(2, 3), keepdim=True)).square()
        variance = squared.sum(dim=(2, 3), keepdim=True) / elements
        attention = torch.sigmoid(squared / (4 * (variance + 1e-4)) + 0.5)
        return self.relu(hidden * attention + self.downsample(values))


class SimAMResNet34(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        widths = (channels, channels * 2, channels * 4, channels * 8)
        counts = (3, 4, 6, 3)
        current = channels
        for index, (width, count) in enumerate(zip(widths, counts, strict=True), 1):
            stride = 1 if index == 1 else 2
            blocks = [SimAMBlock(current, width, stride)]
            blocks.extend(SimAMBlock(width, width) for _ in range(count - 1))
            setattr(self, f"layer{index}", nn.Sequential(*blocks))
            current = width

    def forward(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        hidden = self.relu(self.bn1(self.conv1(values)))
        features = []
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                hidden = block(hidden)
                features.append(hidden)
        return hidden, tuple(features)


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int = 64, acoustic_dim: int = 80) -> None:
        super().__init__()
        feature_dim = channels * 8 * (acoustic_dim // 8)
        self.attention = nn.Sequential(
            nn.Conv1d(feature_dim, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, feature_dim, 1),
            nn.Softmax(dim=2),
        )
        self.out_dim = feature_dim * 2

    def forward(self, values: Tensor) -> Tensor:
        values = values.reshape(values.size(0), -1, values.size(-1))
        weights = self.attention(values)
        mean = torch.sum(values * weights, dim=2)
        std = torch.sqrt(
            (torch.sum(values.square() * weights, dim=2) - mean.square()).clamp(
                min=1e-5
            )
        )
        return torch.cat((mean, std), dim=1)


class TidyVoiceSpeakerModel(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.front = SimAMResNet34()
        self.pooling = AttentiveStatisticsPooling()
        self.bottleneck = nn.Linear(self.pooling.out_dim, embedding_dim)

    def forward(self, fbank: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        hidden, features = self.front(fbank.transpose(1, 2).unsqueeze(1))
        return features, self.bottleneck(self.pooling(hidden))

