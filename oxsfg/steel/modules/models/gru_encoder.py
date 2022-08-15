from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, resnet50, ResNet50_Weights

from torchinfo import summary


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        #encoder = efficientnet_b3(pretrained=True)
        encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        self.encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))

        # freeze the pre-trained weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        print ('-----SUMMARY------')
        summary(self.encoder)

        # maybe throw in a little fc?
        self.fc1 = nn.Linear(2048, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        self.hidden_size=hidden_size

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )

        # self._reset_parameters()
        self.initialize_weights()

    def forward(self, x):

        # encode with VGG features first
        x = torch.stack([
            torch.max(torch.max(self.encoder(t), -1)[0],-1)[0] for t in torch.unbind(x, dim=1)], dim=1) # B x T x Ft
        x = x.squeeze()
        #print ('ENCODER',x.shape)
        x = self.fc1(x)
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
