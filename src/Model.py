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

class CUDNN(nn.Module):
    def __init__(self):
        super(CUDNN, self).__init__()

        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # (b, 64, 14, 14)
            nn.Dropout2d(.25),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # (b, 128, 6, 6)
            nn.Dropout2d(.25),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # (b, 256, 2, 2)
            nn.Dropout2d(.25),
            nn.Conv2d(256, 128, 2),
            nn.ReLU()
            # (b, 128, 1, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
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
    def __init__(self, fe=None, num_mac=4, inter_dim=50):
        super(IRNet, self).__init__()

        # Feature Extractor
        self.feature_extractor = fe
        self.mac_layer = nn.MaxPool2d(num_mac, stride=1)
        self.dim = inter_dim

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2 = x
            feature_x = self.feature_extractor(x1)
            feature_y = self.feature_extractor(x2)
            mvx = self.mac_layer(feature_x).view(-1, self.dim)
            mvy = self.mac_layer(feature_y).view(-1, self.dim)
            mvx = F.normalize(mvx, p=2, dim=1)
            mvy = F.normalize(mvy, p=2, dim=1)

            return mvx, mvy
        else:
            feature = self.feature_extractor(x)
            mac_vector = self.mac_layer(feature)
            mac_vector = mac_vector.view(-1, self.dim)
            mac_vector = F.normalize(mac_vector, p=2, dim=1)

            return mac_vector


