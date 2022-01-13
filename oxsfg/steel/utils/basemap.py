import io

import numpy as np
import requests
from PIL import Image
from shapely import geometry


def basemap_worker(
    pt_wgs84: geometry.Point, zoom_level: int, api_key_path: str, **kwargs
) -> np.ndarray:
    """Wrapper to get single point sample at a given zoom level"""

    with open(api_key_path) as f:
        API_KEY = f.read()

    return get_basemap_arr(pt_wgs84, zoom_level, API_KEY)


def get_basemap_arr(
    pt_wgs84: geometry.Point,
    zoom_level: int,
    API_KEY: str,
) -> np.ndarray:

    urlstr = "https://maps.googleapis.com/maps/api/staticmap?" + "&".join(
        [
            f"""center={pt_wgs84.y},{pt_wgs84.x}""",
            f"""zoom={zoom_level}""",
            """size=400x400""",
            """scale=2""",
            """maptype=satellite""",
            """format=png""",
            f"""key={API_KEY}""",
        ]
    )

    r = requests.get(urlstr, allow_redirects=True)
    # print (r.content)

    image_data = r.content
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")

    return np.asarray(image)


def mosaic_basemap(
    extents_polygon_wgs84: geometry.Polygon,
    zoom_level: int,
    api_key_path: str,
) -> np.ndarray:
    """Get a mosaic of basemap imagery for a chosen zoom level."""

    with open(api_key_path) as f:
        f.read()

    # TODO

    return 1


def basemap_poly(
    poly_wgs84: geometry.Polygon,
    api_key_path: str,
    annotate: bool = False,
    annotate_color: str = "g",
) -> np.ndarray:
    """Choose an appropriate zoom level for a complete polygon and retrieve the basemap. Optionally annotate the polygon."""

    with open(api_key_path) as f:
        API_KEY = f.read()

    zoom_dict = dict(
        zip(
            range(1, 21),
            [
                156543.03392
                * np.cos(poly_wgs84.centroid.y * np.pi / 180)
                / np.power(2, z)
                for z in range(1, 21)
            ],
        )
    )

    box_sides = 5  # TODO: get side length from UTM conversion

    zoom_level = (
        np.max(
            np.argwhere(
                np.array([(zoom_dict[k] * 400 - max(box_sides)) for k in range(1, 21)])
                > 0.0
            )
        )
        + 1
    )

    arr = get_basemap_arr(
        pt_wgs84=poly_wgs84.centroid, zoom_level=zoom_level, API_KEY=API_KEY
    )

    if annotate:
        pass

    return arr
