import glob
import os
import pickle
from datetime import datetime
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from torchvision.models import resnet50, ResNet50_Weights


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


MAX_VALS = {
    "L8": 4000,
    "S2": 4000,
    "L5": 120,
    "L7": 3000,
}
VIS_BANDS = {
    "L8": (3, 2, 1),
    "S2": (3, 2, 1),
    "L5": (2, 1, 0),
    "L7": (2, 1, 0),
}

BANDS_1MIC = dict(zip(["L5", "L7", "L8", "S2"], [4, 4, 5, 10]))
BANDS_2MIC = dict(zip(["L5", "L7", "L8", "S2"], [6, 6, 6, 11]))

BRIGHTNESS = 0.46


class SequenceVizDataset(Dataset):
    """A dataset that prepares image-target samples."""

    def __init__(
        self,
        gdf_path: str,
        data_root: str,
        target_column: str,
        sequence_len: int,
        pretrain_norm: bool,
        resize_dim: Optional[Tuple[int]] = (256, 256),
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
        self.target_column = target_column
        self.sequence_len = sequence_len
        self.resize_dim = resize_dim

        # get records
        print ('RECORDS PATH', os.path.join(data_root, "*", "*.npz"))
        self.records = self.get_records(os.path.join(data_root, "*", "*.npz"))

        # filter for sequence len
        self.records = [
            r for r in self.records if len(r["revisits"]) >= self.sequence_len
        ]
        pickle.dump(self.records,open('./tmp/records.pkl','wb'))

        
        weights = ResNet50_Weights.DEFAULT
        
        
        self.pretrain_norm = pretrain_norm
        self.pretrain_mat = {
            "mean": np.array([0.485, 0.456, 0.406]),
            "std": np.array([0.229, 0.224, 0.225]),
        }
        
        
        self.pretrain_transforms = weights.transforms()

    def prep_vis_arr(self, r):
        """return the array ready for PIL"""
        arr = np.load(r["f"])["arr"][:, :, VIS_BANDS[r["constellation"]]]

        if r["constellation"] == "L5":
            arr = np.stack(
                [arr[:, :, 0] / 224, arr[:, :, 1] / 205, arr[:, :, 2] / 460], axis=-1
            )
        else:
            arr = arr / MAX_VALS[r["constellation"]]

        arr = arr * BRIGHTNESS / arr[arr > 0].mean()

        arr = (arr.clip(0, 1) * 255).astype("uint8")

        return arr

    def get_records(self, glob_pattern):

        records = glob.glob(glob_pattern)

        sites = list({r.split("/")[-2] for r in records})

        records = [
            {"site": site, "revisits": [{"f": f} for f in records if site in f]}
            for site in sites
        ]
        records = [r for r in records if r['site'] in self.gdf.loc[~self.gdf[self.target_column].isna()].index]
        
        print ('TOTAL RECORDS',len(records))

        for record in records:
            for rec in record["revisits"]:
                if os.path.split(rec["f"])[-1][0:4] == "LC08":
                    rec["constellation"] = "L8"
                elif os.path.split(rec["f"])[-1][0:4] == "LT05":
                    rec["constellation"] = "L5"
                elif os.path.split(rec["f"])[-1][0:4] == "LE07":
                    rec["constellation"] = "L7"
                else:
                    rec["constellation"] = "S2"

        for record in records:
            for rec in record["revisits"]:
                if rec["constellation"] in ["L8", "L7", "L5"]:
                    rec["dt"] = datetime.strptime(
                        os.path.split(rec["f"])[-1].split("_")[-1][0:8], "%Y%m%d"
                    )
                else:
                    rec["dt"] = datetime.strptime(
                        os.path.split(rec["f"])[-1][0:8], "%Y%m%d"
                    )

        # sort by datetime
        for record in records:
            record["revisits"] = sorted(record["revisits"], key=lambda r: r["dt"])

        # get nearest neighbours
        for record in records:
            record["revisits"][0]["nn"] = (
                record["revisits"][1]["dt"] - record["revisits"][0]["dt"]
            ).days
            record["revisits"][-1]["nn"] = (
                record["revisits"][-1]["dt"] - record["revisits"][-2]["dt"]
            ).days

            for r, r_n1, r_p1 in zip(
                record["revisits"][1:-1],
                record["revisits"][:-2],
                record["revisits"][2:],
            ):
                r["nn"] = min((r_p1["dt"] - r["dt"]).days, (r["dt"] - r_n1["dt"]).days)

        # get softmax probabilities
        for record in records:
            nn_probs = softmax(np.nan_to_num(np.log10(np.array([rec["nn"] for rec in record["revisits"]])+1)))
            
            if np.any(np.isnan(nn_probs)):
                for ii_r,r in enumerate(record["revisits"]):
                    print (nn_probs[ii_r], r)
                    
                exit()
            
            for ii_r, rec in enumerate(record["revisits"]):
                rec["prob"] = nn_probs[ii_r]

        return records

    def fetch_item(self, r):

        # load and visualise
        arr = self.prep_vis_arr(r)

        # maybe resize
        im = Image.fromarray(arr, "RGB")

        if tuple(arr.shape[1:]) != self.resize_dim:

            im = im.resize(self.resize_dim, Image.ANTIALIAS)

        # return np.array(im)
        return im

    def __getitem__(self, index):

        # get records for chosen index
        site_records = self.records[index]
        
        if np.any(np.isnan([r["prob"] for r in site_records["revisits"]])):
            print ([r for r in site_records["revisits"] if np.isnan(r['prob'])])
            raise ValueError("porb is nan")

        # randomly choose indices
        chosen_idx = sorted(
            np.random.choice(
                len(site_records["revisits"]),
                self.sequence_len,
                replace=False,
                p=[r["prob"] for r in site_records["revisits"]],
            )
        )

        # for record, load
        arrs = []
        for idx in chosen_idx:
            arr = self.fetch_item(site_records["revisits"][idx])

            # do some preprocessing for prtrained models
            if self.pretrain_norm:
                # arr = (arr / 255.0 - self.pretrain_mat["mean"]) / (
                #     self.pretrain_mat["std"]
                # )
                
                arr = np.array(self.pretrain_transforms(arr))
                
            else:
                arr = np.array(arr).transpose(2,0,1)

            arrs.append(arr)

        X = np.stack(arrs)
        # print ('X shape', X.shape)
        # X = np.transpose(X, (0,3,1,2))

        # do Y
        actual_age = self.gdf.loc[site_records["site"], self.target_column]

        Y = (
            pd.to_datetime([site_records["revisits"][idx]["dt"] for idx in chosen_idx])
            >= datetime(int(actual_age), 1, 1)
        ).astype(int)

        return torch.from_numpy(X).float(), torch.from_numpy(Y).float(), np.array([(site_records["revisits"][idx]["dt"]-datetime(1970,1,1)).days for idx in chosen_idx])

    def __len__(self):
        return len(self.records)
