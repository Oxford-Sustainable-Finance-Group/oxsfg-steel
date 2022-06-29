import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = vgg19(pretrained=True)

        # freeze the pre-trained weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512, 24)
        self.fc2 = nn.Linear(24, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):

        # encode with VGG features first
        x = self.encoder(x)
        # x = self.activation['avgpool'].squeeze()
        # x = torch.flatten(x,1)

        # avgpool whole spatial dim
        x = torch.mean(x, dim=(2, 3))
        # x = F.max_pool2d(x, 9)

        # flatten (figure out dimensions?)
        # x = torch.flatten(x, 1)

        # last fully connecteds. Relu for log(R(target))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook
