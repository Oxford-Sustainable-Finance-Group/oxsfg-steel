import click
import geopandas as gpd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from oxsfg.steel.utils import multi_worker_pt, tqdm_joblib


@click.group()
def cli():
    pass


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
    gdf = gpd.read_file(gdf_path).iloc[0:20]

    N_WORKERS = cfg["N_WORKERS"]

    # build the dataset in parallel
    jobs = []

    for idx, row in gdf.iterrows():
        jobs.append(delayed(multi_worker_pt)(idx, row["geometry"], cfg, save_root))

    with tqdm_joblib(
        tqdm(desc=f"deploying GEE on gcp with {N_WORKERS}", total=len(gdf)),
    ):
        Parallel(n_jobs=N_WORKERS, verbose=0, prefer="threads")(jobs)

    return 1


if __name__ == "__main__":
    cli()
