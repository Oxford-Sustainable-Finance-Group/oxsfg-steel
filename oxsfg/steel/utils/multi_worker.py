import os

import numpy as np
from shapely import geometry

from oxsfg.steel.utils import basemap_worker, gee_worker

WORKERS = {
    "BASEMAP": basemap_worker,
    "GEE_WORKER": gee_worker,
}


def multi_worker_pt(
    idx: str,
    geom: geometry.Point,
    cfg: dict,
    save_root: str,
):
    """
    A worker to download a single sample from multiple end points/configs

    """

    sample = dict()

    for kk, vv in cfg["download_specs"].items():

        sample[kk] = WORKERS[vv["worker"]](geom, **vv)

    if save_root[0:5] == "gs://":
        # save to gcs
        raise NotImplementedError
    else:
        np.savez(os.path.join(save_root, f"{idx}.npz"), **sample)
