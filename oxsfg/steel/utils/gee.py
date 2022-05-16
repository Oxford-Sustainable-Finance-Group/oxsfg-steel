import io
import json
import urllib
from datetime import datetime
from typing import List, Optional

import numpy as np
import pyproj
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from pandas import to_datetime
from shapely import geometry, ops

from oxsfg.steel.utils.utils import get_utm_epsg


def gee_worker(
    pt_wgs84: geometry.Point,
    get_single: bool,
    resolution: int,
    patch_size: int,
    start_date_str: str,
    end_date_str: str,
    collection_id: str,
    bands: List[str],
    credentials: str,
    mosaic: bool = True,
    cloud_coverage_filter: Optional[int] = None,
    sort_by: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Get imagery from GEE, optionally mosaicing.

    Parameters
    ----------
    geom: the point geometry centroid
    resolution: the resolution to sample in meters
    patch_size: the size length of the patch
    start_dt: a parseable datetime string for imagery starttime
    end_dt: a parseable datetime string for imagery endtime
    collection_id: the GEE collection_id to query
    bands: a list of band names to query from the collection
    mosaic: optionally obtain imagery mosaicing from multiple ids.
    cloud_coverage_filter: an optional filter to select images meeting certain cloud cover thresholds
    sort_by: one of ['cloud_cover','date']. Sort result images prior to fetching. 'date' is most recent. default is random.


    Returns
    -------
    arr: The (optionally mosaiced) imager of shape [len(bands), patch_size, patch_size]

    """

    if sort_by is not None:
        assert sort_by in [
            "cloud_cover",
            "CLOUD_COVER",
            "CLOUD_COVERAGE_ASSESSMENT",
            "date",
        ], "'sort_by' must be one of ['cloud_cover','CLOUD_COVER','CLOUD_COVERAGE_ASSESSMENT','date']."

    # parse the datetimes
    start_dt = to_datetime(start_date_str).to_pydatetime()
    end_dt = to_datetime(end_date_str).to_pydatetime()

    # build the scoped session
    credentials = service_account.Credentials.from_service_account_file(credentials)
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(scoped_credentials)

    # convert AOI_wgs
    crs_wgs84 = pyproj.CRS("EPSG:4326")
    utm_epsg = get_utm_epsg(pt_wgs84.y, pt_wgs84.x)
    crs_utm = pyproj.CRS(f"EPSG:{utm_epsg}")

    reproject_wgs2utm = pyproj.Transformer.from_crs(
        crs_wgs84, crs_utm, always_xy=True
    ).transform
    reproject_utm2wgs = pyproj.Transformer.from_crs(
        crs_utm, crs_wgs84, always_xy=True
    ).transform

    pt_utm = ops.transform(reproject_wgs2utm, pt_wgs84)
    aoi_utm = geometry.box(
        pt_utm.x - (resolution * patch_size) / 2,
        pt_utm.y - (resolution * patch_size) / 2,
        pt_utm.x + (resolution * patch_size) / 2,
        pt_utm.y + (resolution * patch_size) / 2,
    )
    aoi_wgs = ops.transform(reproject_utm2wgs, aoi_utm)

    # get the ids
    image_metadata = _get_GEE_ids(session, start_dt, end_dt, collection_id, aoi_wgs)
    # print (image_metadata[0])

    # optionally filter them
    if cloud_coverage_filter is not None:
        image_metadata = [
            im
            for im in image_metadata
            if im["properties"][sort_by] < cloud_coverage_filter
        ]

    # sort them
    if sort_by is not None:
        if sort_by == "date":
            image_metadata = sorted(
                image_metadata, key=lambda im: im["endTime"], reverse=True
            )  # reverse for most recent
        elif "cloud" in sort_by.lower():
            image_metadata = sorted(
                image_metadata,
                key=lambda im: im["properties"][sort_by],
            )

    # optionally mosaic the patch

    # get x_offset, y_offset, and crs from shape
    x_off = int(aoi_utm.bounds[0])
    y_off = int(aoi_utm.bounds[3])
    crs_code = f"EPSG:{utm_epsg}"

    if get_single:
        if mosaic:
            # do mosaic and return
            raise NotImplementedError

        else:

            if sort_by is not None:
                image_name = image_metadata[0]["name"]

            else:
                # no sort -> random
                image_name = image_metadata[np.random.choice(len(image_metadata))][
                    "name"
                ]

            return _get_GEE_arr(
                session=session,
                name=image_name,
                bands=bands,
                x_off=x_off,
                y_off=y_off,
                patch_size=patch_size,
                crs_code=crs_code,
            )
    else:
        # return delayed

        delayed_call_data = [
            (session, im_data["name"], bands, x_off, y_off, patch_size, crs_code)
            for im_data in image_metadata
        ]

        return delayed_call_data


def _get_GEE_ids(
    session: AuthorizedSession,
    start_date: datetime,
    end_date: datetime,
    collection_id: str,
    aoi_wgs: geometry.Polygon,
) -> dict:
    """Fetch all the images in a GEE collection between two dates."""

    project = "projects/earthengine-public"

    name = f"{project}/assets/{collection_id}"

    url = "https://earthengine.googleapis.com/v1alpha/{}:listImages?{}".format(
        name,
        urllib.parse.urlencode(
            {
                "startTime": start_date.isoformat() + ".000Z",
                "endTime": end_date.isoformat() + ".000Z",
                "region": json.dumps(geometry.mapping(aoi_wgs)),
            }
        ),
    )

    response = session.get(url)
    content = response.content

    try:
        return json.loads(content)["images"]
    except Exception as e:
        print("Error!")
        print(content)
        raise e


def _get_GEE_arr(
    session: AuthorizedSession,
    name: str,
    bands: List[str],
    x_off: int,
    y_off: int,
    patch_size: int,
    crs_code: str,
) -> np.ndarray:
    """Fetch an ndarray of pixels from the GEE REST API"""

    url = f"https://earthengine.googleapis.com/v1alpha/{name}:getPixels"
    body = json.dumps(
        {
            "fileFormat": "NPY",
            "bandIds": bands,
            "grid": {
                "affineTransform": {
                    "scaleX": 10,
                    "scaleY": -10,
                    "translateX": x_off,
                    "translateY": y_off,
                },
                "dimensions": {"width": patch_size, "height": patch_size},  #
                "crsCode": crs_code,
            },
        }
    )

    pixels_response = session.post(url, body)
    pixels_content = pixels_response.content

    try:
        arr = np.load(io.BytesIO(pixels_content))

        return np.dstack([arr[el] for el in arr.dtype.names]).astype(np.int16)
    except Exception as e:
        print("ERROR!")
        print("url")
        print(url)
        print("body")
        print(body)
        print(pixels_content)
        raise e
