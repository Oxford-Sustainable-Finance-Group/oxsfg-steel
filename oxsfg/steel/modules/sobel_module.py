import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torchinfo import summary

from oxsfg.steel.modules.models import SobelGRU


class SobelModule(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()

        print (hidden_size, input_size)
        print (type(hidden_size), type(input_size))
        self.model = SobelGRU(hidden_size).float()
        self.val_targets = {"y": [], "y_hat": []}

        print("-------- SUMMARY --------")
        summary(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        # y = y.unsqueeze(dim=1)
        y_hat = self(x).squeeze()
        # print (y)
        # print (y_hat)
        #print ('SHAPES',y.shape, y_hat.shape)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("Loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        # y = y.unsqueeze(dim=1)
        # print ('YSHAPE',y.shape)
        y_hat = self(x).squeeze()
        # print ('YHAT_SHAPE',y_hat.shape)
        
        if len(y_hat.shape)!=2:
            y_hat = y_hat.unsqueeze(0) # for batch_size=1 in val
        #print ('SHAPES',y.shape, y_hat.shape)
        loss = F.binary_cross_entropy(y_hat, y)
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

    def predict_step(self, batch, batch_idx, dataset_idx):

        x, y, dates = batch
        y_hat = self(x)

        return {"y_hat": y_hat, "y": y, 'dates':dates}

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=0.02
        )
