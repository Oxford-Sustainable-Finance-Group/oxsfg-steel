from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, resnet50, ResNet50_Weights
import torch.nn.functional as F

from torchinfo import summary


class SobelFilter(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        cuda = torch.device('cuda:0')
        
        self.F_x=torch.stack([torch.from_numpy(np.array([[1, 0, -1],[2,0,-2],[1,0,-1]]))]*3)
        self.F_x = self.F_x.view((3,1,3,3)).float().to(cuda)

        self.F_y=torch.stack([torch.from_numpy(np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]]))]*3)
        self.F_y = self.F_y.view((3,1,3,3)).float().to(cuda)

        
    def forward(self, x):
        # reduce to one grad-magnitude im
        
        G_x= F.conv2d(x, self.F_x, padding=1, groups=3)
        
        G_y= F.conv2d(x, self.F_y, padding=1, groups=3)
        
        G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
        
        return G


class SobelGRU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        #encoder = efficientnet_b3(pretrained=True)
        self.encoder = SobelFilter()

        # freeze the pre-trained weights
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
            
        print ('-----SUMMARY------')
        summary(self.encoder)

        # maybe throw in a little fc?
        # self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        self.hidden_size=hidden_size

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )

        # self._reset_parameters()
        self.initialize_weights()

    def forward(self, x):

        # encode with VGG features first
        x = torch.stack([
            torch.mean(torch.mean(self.encoder(t), -1),-1) for t in torch.unbind(x, dim=1)], dim=1) # B x T x Ft
        x = x.squeeze()
        #print ('ENCODER',x.shape)
        # x = self.fc1(x)
        # stack and maxpool
        #print ('LIN',x.shape)
        x, _ = self.gru(x)
        #print ('GRU',x.shape)
        x = self.fc2(x)
        #print ('FINALE',x.shape)

        return torch.sigmoid(x)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def _reset_parameters(self, initial_forget_bias: Optional[float] = 0.3):
        """Special initialization of certain model weights."""
        if initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[
                self.hidden_size : 2 * self.hidden_size
            ] = initial_forget_bias

    def initialize_weights(self):
        # We are initializing the weights here with Xavier initialisation
        #  (by multiplying with 1/sqrt(n))
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        sqrt_k = np.sqrt(1 / self.hidden_size)
        for parameters in self.gru.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)
