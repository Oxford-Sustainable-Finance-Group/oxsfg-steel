import pytorch_lightning as pl

from oxsfg.steel.datamodules import PanelTargetDataModule
from oxsfg.steel.experiment import ex
from oxsfg.steel.modules import CNNModule


@ex.automain
def main(
    dataset_parameters: dict,
    model_parameters: dict,
    training_parameters: dict,
) -> int:

    # build data module
    dm = PanelTargetDataModule(dataset_parameters, training_parameters)

    dm.setup()

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    dm.test_dataloader()

    # build model
    module = CNNModule(**model_parameters)

    # train
    trainer = pl.Trainer()
    trainer.fit(module, train_dl, val_dl)

    return 1
