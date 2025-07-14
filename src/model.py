import torch.nn as nn
import torch.nn.functional as F

class CNNExperiment(nn.Module):
    def _init_(self):
        super(CNNExperiment, self)._init_()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)

        return x