import pathlib

import pyminiply
import typer
import viser
from loguru import logger


def show(point_cloud_file: pathlib.Path) -> None:
    points, *_, colors = pyminiply.read(
        point_cloud_file,
        read_normals=False,
        read_uv=False,
    )

    logger.info(f"Loaded {len(points)} points")

    server = viser.ViserServer()

    server.scene.add_point_cloud(
        "point_cloud",
        points=points,
        colors=colors,
        point_size=0.0002,
        point_shape="circle",
    )

    server.sleep_forever()


if __name__ == "__main__":
    typer.run(show)
