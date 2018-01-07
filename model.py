'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.conv3 = nn.Conv2d(40, 80, 4)
        self.fc1   = nn.Linear(80*5*5, 1024)
        self.fc2   = nn.Linear(1024, 2)

    def forward(self, x):
        out = F.max_pool2d(x, 2)
        out = F.relu(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out