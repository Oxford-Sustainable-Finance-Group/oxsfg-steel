import click
import geopandas as gpd

from oxsfg.steel.utils.make_video import make_video


@click.group()
def cli():
    pass


@cli.command()
@click.argument("gdf_path")
@click.argument("data_remote_root")
@click.argument("data_local_root")
@click.argument("image_root")
@click.argument("final_local_path")
@click.argument("final_remote_path")
def generate_videos(
    gdf_path: str,
    data_remote_root: str,
    data_local_root: str,
    image_root: str,
    final_local_path: str,
    final_remote_path: str,
) -> int:

    gdf = gpd.read_file("./data/oxsfg_steel.gpkg").set_index("uid")

    for idx, _row in gdf.loc[~gdf["year"].isna()].iloc[:15].iterrows():

        make_video(
            idx,
            data_remote_root,
            data_local_root,
            image_root,
            final_local_path,
            final_remote_path,
        )


if __name__ == "__main__":
    cli()
