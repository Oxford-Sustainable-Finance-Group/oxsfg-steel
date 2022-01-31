import os
import pickle
from typing import List, Optional

import geopandas as gpd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class PanelTargetDataset(Dataset):
    """A dataset that prepares image-target samples."""

    def __init__(
        self,
        gdf_path: str,
        data_root: str,
        target_column: str,
        data_keys: List[str],
        is_categorical: bool = False,
        zscore_path: Optional[str] = None,
        **kwargs,
    ):
        """Tile dataset
        Args:
            gdf_path (str):
            data_path (str): the root of the data directory
            target_column (str): the target column for prediction
            zscore_path (str, optional): the path to the zscore data dict
        """
        super().__init__()

        self.gdf = gpd.read_file(gdf_path).set_index("uid")
        self.data_root = data_root
        self.is_categorical = is_categorical
        self.data_keys = data_keys
        self.target_column = target_column

        self.gdf = self.gdf.loc[~self.gdf[target_column].isna()]
        self.indices = self.gdf.index.values.tolist()

        # check exists
        for idx in tqdm(self.indices, desc="Check data exists"):
            assert os.path.exists(
                os.path.join(self.data_root, f"{idx}.npz")
            ), f"Missing file: {os.path.join(self.data_root,f'{idx}.npz')}"

        if zscore_path is not None:
            #
            self.zscore_data = self.maybe_get_zscore(zscore_path)

        else:
            self.zscore_data = None

    def load_item(
        self,
        idx: str,
    ) -> np.ndarray:

        f = np.load(os.path.join(self.data_root, f"{idx}.npz"))
        data = {kk: f[kk] for kk in self.data_keys}

        return data

    def maybe_get_zscore(
        self,
        zscore_path: str,
    ) -> np.ndarray:

        if not os.path.exists(zscore_path):
            # making score
            means = {kk: [] for kk in self.data_keys}
            stds = {kk: [] for kk in self.data_keys}

            for idx in tqdm(self.indices, desc="Build ZScore"):
                data = self.load_item(idx)
                for kk in self.data_keys:

                    means[kk].append(data[kk].mean(axis=(1, 2)))
                    stds[kk].append(data[kk].std(axis=(1, 2)))

            zscore_data = {
                kk: {
                    "mean": np.vstack(means[kk]).mean(axis=0),
                    "std": np.vstack(stds[kk]).mean(axis=0),
                }
                for kk in self.data_keys
            }

            pickle.dump(open(zscore_path, "wb"))

        else:
            zscore_data = pickle.load(open(zscore_path, "rb"))

        return zscore_data

    def __getitem__(self, index):

        X = self.load_item(self.indices[index])

        if self.zscore_data is not None:
            for kk in self.data_keys:
                X[kk] = (X[kk] - self.zscore_data[kk]["mean"]) / self.zscore_data[kk][
                    "std"
                ]

        return X

    def __len__(self):
        return len(self.gdf)
