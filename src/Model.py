from torch import nn
from torch.nn import init


# Convolution Auto Encoder
class LeNetAE28(nn.Module):
    def __init__(self):
        super(LeNetAE28, self).__init__()

        # Encoder Network
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
