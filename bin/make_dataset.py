import io

import click
import geopandas as gpd
import pandas as pd
import requests
import yaml
from joblib import Parallel, delayed
from shapely import wkt
from tqdm import tqdm

from oxsfg.steel.utils import multi_worker_pt_multi, multi_worker_pt_single, tqdm_joblib


@click.group()
def cli():
    pass


@cli.command()
@click.argument("version_str")
@click.argument("git_username")
@click.argument("git_token")
@click.argument("outpath")
@click.option("--driver", default="GPKG")
def fetch_geodataframe_from_github(
    version_str: str,
    git_username: str,
    git_token: str,
    outpath: str,
    driver: str,
) -> int:
    """
    Command-line utility to build a local geopandas GeoDataFrame for a dataset version.

    Args
    ----
        version_str: The version of the data to retrieve
        git_username: Your github username
        git_token: Your github access token
        outpath: The savepath for the geodataframe
        driver: The fiona driver to use (optional, default: 'GPKG')

    Returns:
        1
    """

    session = requests.Session()
    session.auth = (git_username, git_token)

    # read as a pandas dataframe, e.g. points dataframe
    data_url = f"https://raw.githubusercontent.com/spatial-finance/oxsfg-asset-level-data/main/steel/{version_str}/steel-points.csv"
    data_io = session.get(data_url).content
    df = pd.read_csv(io.StringIO(data_io.decode("utf-8")))

    # (optionally) cast to geodataframe
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf.to_file(outpath, driver=driver)

    return 1


@cli.command()
@click.argument("gdf_path")
@click.argument("cfg_path")
@click.argument("save_root")
def make_pointcentre_dataset(
    gdf_path: str,
    cfg_path: str,
    save_root: str,
) -> int:
    """
    Command-line utility to build a dataset, prep it for training.

    Args
    ----
        gdf_path: A path to a geopandas dataframe
        cfg_path: Path to a cfg file
        save_root: The savepath for the data

    Returns:
        1
    """

    # load the cfg and the gdf
    cfg = yaml.load(open(cfg_path), Loader=yaml.SafeLoader)
    gdf = gpd.read_file(gdf_path)
    gdf = gdf.set_index("uid")

    N_WORKERS = cfg["N_WORKERS"]

    # build the dataset in parallel
    jobs = []

    if cfg["get_single"]:

        for idx, row in gdf.iterrows():
            jobs.append(
                delayed(multi_worker_pt_single)(idx, row["geometry"], cfg, save_root)
            )

        with tqdm_joblib(
            tqdm(desc=f"deploying GEE on gcp with {N_WORKERS}", total=len(gdf)),
        ):
            Parallel(n_jobs=N_WORKERS, verbose=0, prefer="threads")(jobs)

    else:
        for idx, row in gdf.iloc[2:10].iterrows():
            # get the distributed jobs

            jobs = []
            keyed_args = multi_worker_pt_single(idx, row["geometry"], cfg, save_root)
            for _key, arg_list in keyed_args.items():
                for args in arg_list:
                    jobs.append(delayed(multi_worker_pt_multi)(idx, args, save_root))

            with tqdm_joblib(
                tqdm(
                    desc=f"deploying GEE for {idx} on gcp with {N_WORKERS}",
                    total=len(jobs),
                ),
            ):
                Parallel(n_jobs=N_WORKERS, verbose=0, prefer="threads")(jobs)

    return 1


if __name__ == "__main__":
    cli()
