import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torchinfo import summary

from oxsfg.steel.modules.models import GRUEncoder


class GRUModule(pl.LightningModule):
    def __init__(self, data_key: str, channels_last: bool, **kwargs):
        super().__init__()

        self.model = GRUEncoder().float()
        self.channels_last = channels_last
        self.val_targets = {"y": [], "y_hat": []}

        print("-------- SUMMARY --------")
        summary(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.unsqueeze(dim=1)
        y_hat = self(x)
        # print (y)
        # print (y_hat)
        loss = F.cross_entropy(y_hat, y)
        self.log("Loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y = y.unsqueeze(dim=1)
        y_hat = self(x).view(-1, 1)
        loss = F.cross_entropy(y_hat, y)
        self.log("Loss/val", loss)

        # print (y_hat.shape, y.shape)
        self.val_targets["y"].append(y.cpu().numpy())
        self.val_targets["y_hat"].append(y_hat.cpu().numpy())
        return {"val_loss": loss}

    def on_validation_epoch_end(self):

        R_squared = r2_score(
            np.concatenate(self.val_targets["y"]),
            np.concatenate(self.val_targets["y_hat"]),
        )

        self.log("R_sq", R_squared)

        self.val_targets = {"y": [], "y_hat": []}

        return 1

    def predict_step(self, batch):

        x, y = batch
        y_hat = self(x)

        return {"y_hat": y_hat, "y": y}

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=0.02
        )
