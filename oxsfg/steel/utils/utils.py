import contextlib

import joblib


def get_utm_zone(lat, lon):
    """A function to grab the UTM zone number for any lat/lon location"""
    zone_str = str(int((lon + 180) / 6) + 1)

    if (lat >= 56.0) & (lat < 64.0) & (lon >= 3.0) & (lon < 12.0):
        zone_str = "32"
    elif (lat >= 72.0) & (lat < 84.0):
        if (lon >= 0.0) & (lon < 9.0):
            zone_str = "31"
        elif (lon >= 9.0) & (lon < 21.0):
            zone_str = "33"
        elif (lon >= 21.0) & (lon < 33.0):
            zone_str = "35"
        elif (lon >= 33.0) & (lon < 42.0):
            zone_str = "37"

    return zone_str


def get_utm_epsg(lat, lon):
    zone_str = get_utm_zone(lat, lon)

    if lat > 0:
        return f"326{zone_str}"
    else:
        return f"327{zone_str}"


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
