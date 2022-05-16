import contextlib

import joblib
from google.cloud import storage


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


def blob_exists(f):

    storage_client = storage.Client()
    bucket_name = f.split("/")[0]
    name = "/".join(f.split("/")[1:])
    bucket = storage_client.bucket(bucket_name)
    return storage.Blob(bucket=bucket, name=name).exists(storage_client)


def upload_blob(source_directory: str, target_directory: str):
    """Function to save file to a bucket.
    Args:
        target_directory (str): Destination file path.
        source_directory (str): Source file path
    Returns:
        None: Returns nothing.
    Examples:
        >>> target_directory = 'target/path/to/file/.pkl'
        >>> source_directory = 'source/path/to/file/.pkl'
        >>> save_file_to_bucket(target_directory)
    """

    client = storage.Client()

    bucket_id = target_directory.split("/")[0]
    file_path = "/".join(target_directory.split("/")[1:])

    bucket = client.get_bucket(bucket_id)

    # get blob
    blob = bucket.blob(file_path)

    # upload data
    blob.upload_from_filename(source_directory)

    return target_directory
