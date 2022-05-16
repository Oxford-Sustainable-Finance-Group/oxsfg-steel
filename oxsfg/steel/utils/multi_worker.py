import os

import numpy as np
from shapely import geometry

from oxsfg.steel.utils.basemap import basemap_worker
from oxsfg.steel.utils.gee import _get_GEE_arr, gee_worker
from oxsfg.steel.utils.previewers import basemap_previewer, s2_previewer
from oxsfg.steel.utils.utils import blob_exists, upload_blob

WORKERS = {
    "BASEMAP": basemap_worker,
    "GEE": gee_worker,
}
PREVIEWERS = {
    "S2": s2_previewer,
    "BASEMAP": basemap_previewer,
}


def multi_worker_pt_single(
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

        sample[kk] = WORKERS[vv["worker"]](geom, cfg["get_single"], **vv)

        if vv["preview"] is not None:
            PREVIEWERS[vv["preview"]](
                sample[kk], os.path.join(save_root, f"{idx}-{kk}.png"), **vv
            )

    if cfg["get_single"]:

        if save_root[0:5] == "gs://":
            # save to gcs
            fpath = os.path.join(os.getcwd(), "tmp", f"{idx}.npz")
            np.savez(fpath, **sample)

            upload_blob(fpath, os.path.join(save_root[5:], f"{idx}.npz"))
            os.remove(fpath)

        else:
            np.savez(os.path.join(save_root, f"{idx}.npz"), **sample)

        return 1

    else:
        return sample


# wrap with delayed
def multi_worker_pt_multi(
    idx: str,
    call_args: list,
    save_root: str,
):

    GEE_ID = os.path.split(call_args[1])[-1]

    # check exists first
    if save_root[0:5] == "gs://":
        blob_path = os.path.join(save_root[5:], f"{idx}", f"{GEE_ID}.npz")
        exists = blob_exists(blob_path)
    else:
        exists = os.path.exists(os.path.join(save_root, f"{idx}-{GEE_ID}.npz"))

    if exists:
        return 1

    arr = _get_GEE_arr(*call_args)

    if save_root[0:5] == "gs://":
        # save to gcs
        fpath = os.path.join(os.getcwd(), "tmp", f"{idx}-{GEE_ID}.npz")
        np.savez(fpath, arr=arr)

        upload_blob(fpath, os.path.join(save_root[5:], f"{idx}", f"{GEE_ID}.npz"))
        os.remove(fpath)

    else:
        np.savez(os.path.join(save_root, f"{idx}-{GEE_ID}.npz"), arr=arr)

    return 1
