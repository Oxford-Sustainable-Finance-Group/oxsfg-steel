import os
import pickle
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import torch
from skimage.transform import resize
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
        resize_dim: Optional[Dict[str, list]] = None,
        crop: Optional[dict] = None,
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
        self.crop = crop
        self.resize_dim = resize_dim

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

                    means[kk].append(data[kk].mean(axis=(0, 1)))
                    stds[kk].append(data[kk].std(axis=(0, 1)))

            for kk in self.data_keys:
                print("shape", np.vstack(means[kk]).shape)

            zscore_data = {
                kk: {
                    "mean": np.vstack(means[kk]).mean(axis=0),
                    "std": np.vstack(stds[kk]).mean(axis=0),
                }
                for kk in self.data_keys
            }

            pickle.dump(zscore_data, open(zscore_path, "wb"))

        else:
            zscore_data = pickle.load(open(zscore_path, "rb"))

        return zscore_data

    def _maybe_resize(self, X):

        if self.resize_dim is not None:
            for kk in self.data_keys:
                if kk in self.resize_dim.keys():
                    X[kk] = resize(X[kk], tuple(self.resize_dim[kk]))

        return X

    def _maybe_crop(self, X):

        if self.crop is not None:
            for kk in self.data_keys:
                if kk in self.crop.keys():

                    X[kk] = X[kk][
                        self.crop[kk][0] : -self.crop[kk][2],
                        self.crop[kk][1] : -self.crop[kk][3],
                        :,
                    ]

        return X

    def __getitem__(self, index):

        X = self.load_item(self.indices[index])

        if self.zscore_data is not None:
            for kk in self.data_keys:
                X[kk] = (X[kk] - self.zscore_data[kk]["mean"]) / self.zscore_data[kk][
                    "std"
                ]

        self._maybe_crop(X)
        self._maybe_resize(X)

        for kk in self.data_keys:
            X[kk] = torch.from_numpy(X[kk]).float()

        Y = torch.from_numpy(
            np.array(self.gdf.loc[self.indices[index], self.target_column])
        ).float()

        return X, Y

    def __len__(self):
        return len(self.gdf)
