import pathlib

import pyminiply
import torch
import typer
import viser
from loguru import logger

from dataset import normalize, to_size


def mask(point_cloud: torch.Tensor) -> torch.Tensor:
    point_cloud_masked = point_cloud.clone()
    point_cloud_masked[:, point_cloud_masked[0, :] < 0.5] = 0.0
    return point_cloud_masked


def show(point_cloud_file: pathlib.Path) -> None:
    points, *_ = pyminiply.read(
        point_cloud_file,
        read_normals=False,
        read_uv=False,
    )

    points = torch.from_numpy(points).T
    points = normalize(to_size(points, 100_000))
    # points_masked = mask(points)

    colors = torch.zeros_like(points)

    logger.info(f"Loaded {len(points)} points")

    server = viser.ViserServer()
    server.scene.add_point_cloud(
        "point_cloud",
        points=points.T.numpy(),
        colors=colors.T.numpy(),
        point_size=0.002,
        point_shape="circle",
    )

    # server.scene.add_point_cloud(
    #     "point_cloud_masked",
    #     points=points_masked.T.numpy(),
    #     colors=colors.T.numpy(),
    #     point_size=0.002,
    #     point_shape="circle",
    #     position=(-1.0, -1.0, 0.0),
    # )

    model = torch.load("./models/point_cloud_vae.pt", weights_only=False)
    model.eval()

    with torch.no_grad():
        points_reconstructed = model.forward(points.unsqueeze(0)).squeeze(0)

    print(points_reconstructed)

    server.scene.add_point_cloud(
        "point_cloud_reconstructed",
        points=points_reconstructed.T.numpy(),
        colors=colors.T.numpy(),
        point_size=0.002,
        point_shape="circle",
        position=(-2.0, -2.0, 0.0),
    )

    server.sleep_forever()


if __name__ == "__main__":
    typer.run(show)
