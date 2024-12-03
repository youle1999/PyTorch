import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Output: 32x26x26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Output: 64x11x11
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjust to 64*5*5
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Output: 32x13x13
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Output: 64x5x5
        x = x.view(-1, 64 * 5 * 5)  # Adjusted size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

