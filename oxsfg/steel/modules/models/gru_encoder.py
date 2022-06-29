from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.encoder = efficientnet_b3(pretrained=True)

        # freeze the pre-trained weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # maybe throw in a little fc?
        # self.fc1 = nn.Linear(512, 24)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )

        self._reset_parameters()
        self.initialize_weights()

    def forward(self, x):

        # encode with VGG features first
        x = [self.encoder(t) for t in torch.unbind(x, dim=0)]

        # stack and maxpool
        x = torch.max(torch.stack(x), dim=(2, 3))
        x = self.gru(x)
        x = torch.sigmoif(self.fc2(x))

        return x

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
        for parameters in self.lstm.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)
