import pickle

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from oxsfg.steel.datamodules import PanelTargetDataModule
from oxsfg.steel.experiment import ex
from oxsfg.steel.modules import CNNModule

# from oxsfg.steel.metrics import plot_sample_errors


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
    test_dl = dm.test_dataloader()

    # build model
    module = CNNModule(**model_parameters)

    name = module.model.encoder._get_name()

    tb_logger = TensorBoardLogger("experiments/tensorboard", name=name)

    # train
    trainer = pl.Trainer(
        logger=tb_logger,
        log_every_n_steps=5,
        max_epochs=training_parameters["max_epochs"],
        gpus=training_parameters["gpus"],
    )
    trainer.fit(
        module,
        train_dl,
        val_dl,
    )

    predictions = trainer.predict(dataloaders=[val_dl, test_dl])

    pickle.dump(predictions, open("./tmp/predictions.pkl", "wb"))
    ex.add_artifact("./tmp/predictions.pkl")

    return 1
