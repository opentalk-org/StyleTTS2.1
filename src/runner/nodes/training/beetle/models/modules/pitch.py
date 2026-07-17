import torch
from torch import Tensor, nn


class ResBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        leaky_relu_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.downsample = input_channels != output_channels
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        )
        self.conv1by1 = (
            nn.Conv2d(input_channels, output_channels, 1, bias=False)
            if self.downsample
            else nn.Identity()
        )

    def forward(self, features: Tensor) -> Tensor:
        features = self.pre_conv(features)
        return self.conv(features) + self.conv1by1(features)


class JDCNet(nn.Module):
    def __init__(
        self,
        num_class: int = 722,
        seq_len: int = 31,
        leaky_relu_slope: float = 0.01,
    ) -> None:
        super().__init__()
        del seq_len
        self.num_class = num_class
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )
        self.res_block1 = ResBlock(64, 128)
        self.res_block2 = ResBlock(128, 192)
        self.res_block3 = ResBlock(192, 256)
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.2),
        )
        self.bilstm_classifier = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_detector = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(512, self.num_class)
        self.detector = nn.Linear(512, 2)
        self.apply(self._initialize_weights)

    def get_feature_gan(self, features: Tensor) -> Tensor:
        features = features.float().transpose(-1, -2)
        features = self.conv_block(features)
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.res_block3(features)
        features = self.pool_block[0](features)
        features = self.pool_block[1](features)
        return features.transpose(-1, -2)

    def get_feature(self, features: Tensor) -> Tensor:
        features = features.float().transpose(-1, -2)
        features = self.conv_block(features)
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.res_block3(features)
        features = self.pool_block[0](features)
        features = self.pool_block[1](features)
        return self.pool_block[2](features)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        sequence_length = features.shape[-1]
        features = features.float().transpose(-1, -2)
        features = self.conv_block(features)
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.res_block3(features)
        pooled = self.pool_block[0](features)
        pooled = self.pool_block[1](pooled)
        gan_features = pooled.transpose(-1, -2)
        pooled = self.pool_block[2](pooled)
        classified = pooled.permute(0, 2, 1, 3).contiguous()
        classified = classified.view(-1, sequence_length, 512)
        classified, _ = self.bilstm_classifier(classified)
        classified = self.classifier(classified.contiguous().view(-1, 512))
        classified = classified.view(-1, sequence_length, self.num_class)
        return torch.abs(classified), gan_features, pooled

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, (nn.LSTM, nn.LSTMCell)):
            for parameter in module.parameters():
                if len(parameter.shape) >= 2:
                    nn.init.orthogonal_(parameter.data)
                else:
                    nn.init.normal_(parameter.data)
