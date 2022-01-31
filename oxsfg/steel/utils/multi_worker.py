import os

import numpy as np
from shapely import geometry

from oxsfg.steel.utils.basemap import basemap_worker
from oxsfg.steel.utils.gee import gee_worker
from oxsfg.steel.utils.previewers import basemap_previewer, s2_previewer

WORKERS = {
    "BASEMAP": basemap_worker,
    "GEE": gee_worker,
}
PREVIEWERS = {
    "S2": s2_previewer,
    "BASEMAP": basemap_previewer,
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

        if vv["preview"] is not None:
            PREVIEWERS[vv["preview"]](
                sample[kk], os.path.join(save_root, f"{idx}-{kk}.png"), **vv
            )

    if save_root[0:5] == "gs://":
        # save to gcs
        raise NotImplementedError
    else:
        np.savez(os.path.join(save_root, f"{idx}.npz"), **sample)
