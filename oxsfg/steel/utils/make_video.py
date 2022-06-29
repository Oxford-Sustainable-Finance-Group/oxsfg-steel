import glob
import os
import subprocess
from datetime import datetime

import numpy as np
from loguru import logger
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

from oxsfg.steel.utils.utils import upload_blob

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
FONTS = {
    "L8": ImageFont.truetype("DejaVuSans.ttf", 9),
    "S2": ImageFont.truetype("DejaVuSans.ttf", 12),
    "L5": ImageFont.truetype("DejaVuSans.ttf", 6),
    "L7": ImageFont.truetype("DejaVuSans.ttf", 9),
}
BRIGHTNESS = 0.46

FONTS = {
    "L8": ImageFont.truetype("DejaVuSans.ttf", 9),
    "S2": ImageFont.truetype("DejaVuSans.ttf", 12),
    "L5": ImageFont.truetype("DejaVuSans.ttf", 6),
    "L7": ImageFont.truetype("DejaVuSans.ttf", 9),
}

hot = cm.hot


def prep_swir(r, dd):
    """return the array ready for PIL"""
    arr = np.load(r["f"])["arr"][:, :, dd[r["constellation"]]]

    arr = (arr - arr[arr > 0].mean()) / arr[arr > 0].std()

    arr = hot((arr + 3) / 6)

    arr = arr[:, :, 0:3]

    arr = (arr.clip(0, 1) * 255).astype("uint8")

    return arr


def prep_vis_arr(r):
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


def parse_records(glob_pattern):

    records = glob.glob(glob_pattern)

    records = [
        {
            "f": f,
        }
        for f in records
    ]

    for rec in records:
        if os.path.split(rec["f"])[-1][0:4] == "LC08":
            rec["constellation"] = "L8"
        elif os.path.split(rec["f"])[-1][0:4] == "LT05":
            rec["constellation"] = "L5"
        elif os.path.split(rec["f"])[-1][0:4] == "LE07":
            rec["constellation"] = "L7"
        else:
            rec["constellation"] = "S2"

    for rec in records:
        if rec["constellation"] in ["L8", "L7", "L5"]:
            rec["dt"] = datetime.strptime(
                os.path.split(rec["f"])[-1].split("_")[-1][0:8], "%Y%m%d"
            )
        else:
            rec["dt"] = datetime.strptime(os.path.split(rec["f"])[-1][0:8], "%Y%m%d")

    return records


def download_data(bucket_name, data_remote_pattern, data_root_local):

    download_command = [
        "gsutil",
        "-m",
        "cp",
        f"gs://{bucket_name}/{data_remote_pattern}",
        data_root_local,
    ]
    subprocess.run(download_command, capture_output=True)

    return 1


def call_ffmpeg(image_root, final_vid_path):

    # make all three videos
    for view in ["vis", "1mic", "2mic"]:
        im_pattern = os.path.join(image_root, view, "%04d.jpg")
        out_vid = f"{view}.mp4"
        subprocess.run(
            ["ffmpeg", "-framerate", "10", "-i", im_pattern, "-r", "5", "-y", out_vid],
            capture_output=True,
        )

    # rescale the biggy boi

    subprocess.run(
        ["ffmpeg", "-i", "vis.mp4", "-vf", "scale=-1:512", "-y", "vis_big.mp4"],
        capture_output=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            "1mic.mp4",
            "-i",
            "2mic.mp4",
            "-filter_complex",
            "vstack=inputs=2",
            "-y",
            "mic_2panel.mp4",
        ],
        capture_output=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            "vis_big.mp4",
            "-i",
            "mic_2panel.mp4",
            "-filter_complex",
            "hstack=inputs=2",
            "-y",
            f"{final_vid_path}",
        ],
        capture_output=True,
    )


def records2ims_swir(records, image_root, bands_dict, view):
    for ii, rec in enumerate(sorted(records, key=lambda rec: rec["dt"])):
        arr = prep_swir(rec, bands_dict)
        # print (arr.min(), arr.max(),rec['constellation'])
        im = Image.fromarray(arr, "RGB")

        if rec["constellation"] in ["L5", "L7", "L8"]:
            im = im.resize((256, 256), Image.ANTIALIAS)

        draw = ImageDraw.Draw(im)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text(
            (0, 0),
            rec["dt"].isoformat()[0:10] + f' - {rec["constellation"]}',
            (255, 255, 255),
            font=FONTS["S2"],
        )  # fonts[rec['constellation']])
        draw.text(
            (128, 128), "+", (255, 255, 255), font=FONTS["S2"]
        )  # fonts[rec['constellation']])
        # fout = os.path.join(root,'data','all_imagery_staging',rec['dt'].isoformat()[0:10]+f'-{rec["constellation"]}.jpg')
        fout = os.path.join(image_root, view, f"{ii:04d}.jpg")

        im.save(fout)


def records2ims(records, image_root):

    for ii, rec in enumerate(sorted(records, key=lambda rec: rec["dt"])):
        arr = prep_vis_arr(rec)
        # print (arr.min(), arr.max(),rec['constellation'])
        im = Image.fromarray(arr, "RGB")

        if rec["constellation"] in ["L5", "L7", "L8"]:
            im = im.resize((256, 256), Image.ANTIALIAS)

        draw = ImageDraw.Draw(im)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text(
            (0, 0),
            rec["dt"].isoformat()[0:10] + f' - {rec["constellation"]}',
            (255, 255, 255),
            font=FONTS["S2"],
        )  # fonts[rec['constellation']])
        draw.text(
            (128, 128), "+", (255, 255, 255), font=FONTS["S2"]
        )  # fonts[rec['constellation']])
        # fout = os.path.join(root,'data','all_imagery_staging',rec['dt'].isoformat()[0:10]+f'-{rec["constellation"]}.jpg')
        fout = os.path.join(image_root, "vis", f"{ii:04d}.jpg")

        im.save(fout)


def make_video(
    idx,
    data_remote_root,
    data_local_root,
    image_root,
    final_local_path,
    final_remote_path,
):

    # check some routes

    # download all the data
    logger.info(f"GENERATING VIDEOS: {idx}")
    bucket_name = data_remote_root.split("/")[0]
    remote_pattern = os.path.join(*data_remote_root.split("/")[1:], idx, "*.npz")
    logger.info(f"Downloading Data: {remote_pattern} to {data_local_root}")
    download_data(bucket_name, remote_pattern, data_local_root)

    # parse the records
    logger.info("Parsing records")
    glob_pattern = os.path.join(data_local_root, "*.npz")
    records = parse_records(glob_pattern)

    # make the images
    logger.info("Making Images: VIS")
    records2ims(records, image_root)
    logger.info("Making Images: 1MIC")
    records2ims_swir(records, image_root, BANDS_1MIC, "1mic")
    logger.info("Making Images: 2MIC")
    records2ims_swir(records, image_root, BANDS_2MIC, "2mic")

    # make the videos - ffmpeg
    logger.info("Making Videos")
    final_vid_path = os.path.join(final_local_path, f"{idx}.mp4")
    call_ffmpeg(image_root, final_vid_path)

    # push the video
    logger.info("Pushing the final video")
    upload_blob(final_vid_path, os.path.join(final_remote_path, f"{idx}.mp4"))

    # clear the data, images and push the video
    # clear raw
    logger.info("Clearning data - raw")
    fs = glob.glob(os.path.join(data_local_root, "*.npz"))
    for f in fs:
        if os.path.isfile(f):
            os.remove(f)

    # clear , vis, 1mic, 2mic
    for view in ["vis", "1mic", "2mic"]:
        logger.info(f"Cleaning data - {view}")
        fs = glob.glob(os.path.join(image_root, view, "*.jpg"))
        for f in fs:
            if os.path.isfile(f):
                os.remove(f)

    # clear videos
    logger.info("Clearning data - intermediate vids")
    for title in ["vis", "vis_big", "1mic", "2mic", "mic_2panel"]:
        os.remove(f"{title}.mp4")

    logger.info(f"DONE {idx}!")

    return 1
