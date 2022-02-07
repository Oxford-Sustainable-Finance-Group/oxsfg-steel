from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from oxsfg.steel.datamodules.datasets import PanelTargetDataset


class PanelTargetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_parameters: dict, training_parameters: dict):
        super().__init__()

        for key in [
            "gdf_path",
            "data_root",
            "target_column",
            "data_keys",
            "is_categorical",
            "zscore_path",
            "crop",
            "resize_dim",
        ]:
            setattr(self, key, dataset_parameters[key])

        for key in ["batch_size", "val_split", "test_split"]:
            setattr(self, key, training_parameters[key])

    def setup(self, stage: Optional[str] = None):

        ds_full = PanelTargetDataset(
            gdf_path=self.gdf_path,
            data_root=self.data_root,
            target_column=self.target_column,
            data_keys=self.data_keys,
            is_categorical=self.is_categorical,
            zscore_path=self.zscore_path,
            crop=self.crop,
            resize_dim=self.resize_dim,
        )

        val_size = int(len(ds_full) * self.val_split)
        test_size = int(len(ds_full) * self.test_split)
        trn_size = len(ds_full) - val_size - test_size

        self.ds_train, self.ds_val, self.ds_test = random_split(
            ds_full, [trn_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass
