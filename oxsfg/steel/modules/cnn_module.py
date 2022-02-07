import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchinfo import summary

from oxsfg.steel.modules.models import CNNEncoder


class CNNModule(pl.LightningModule):
    def __init__(self, data_key: str, channels_last: bool, **kwargs):
        super().__init__()

        self.data_key = data_key
        self.model = CNNEncoder().float()
        self.channels_last = channels_last

        print("-------- SUMMARY --------")
        summary(self.model)

    def forward(self, x):

        if self.channels_last:
            # swap channels
            x = x[self.data_key].permute(0, 3, 1, 2)
        else:
            x = x[self.data_key]

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).unsqueeze()
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=0.02
        )
