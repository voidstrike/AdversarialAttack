from torch import nn
from torch.nn import init
import torch.nn.functional as F


# Convolution Auto Encoder
class LeNetAE28(nn.Module):
    def __init__(self):
        super(LeNetAE28, self).__init__()

        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 20, 5, stride=1),  # (b, 20, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (b, 20, 12, 12)
            nn.Conv2d(20, 50, 5, stride=1),  # (b, 50, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # (b, 50, 4, 4)
        )

        self.fc = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.flatten(start_dim=1)
        res = self.fc(feature)
        return res


class IRNet(nn.Module):
    def __init__(self):
        super(IRNet, self).__init__()

        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 20, 5, stride=1),  # (b, 20, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (b, 20, 12, 12)
            nn.Conv2d(20, 50, 5, stride=1),  # (b, 50, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # (b, 50, 4, 4)
        )

        self.mac_layer = nn.MaxPool2d(4, stride=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2 = x
            feature_x = self.feature_extractor(x1)
            feature_y = self.feature_extractor(x2)
            mvx = self.mac_layer(feature_x).view(-1, 50)
            mvy = self.mac_layer(feature_y).view(-1, 50)
            mvx = F.normalize(mvx, p=2, dim=1)
            mvy = F.normalize(mvy, p=2, dim=1)

            return mvx, mvy
        else:
            feature = self.feature_extractor(x)
            mac_vector = self.mac_layer(feature)
            mac_vector = mac_vector.view(-1, 50)
            mac_vector = F.normalize(mac_vector, p=2, dim=1)

            return mac_vector


