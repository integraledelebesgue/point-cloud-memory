import pathlib

import numpy
import pyminiply
import torch
import typer
import viser
from loguru import logger

import datasets

torch.manual_seed(0)


def load_point_cloud(location: pathlib.Path, num_points: int) -> torch.Tensor:
    points, *_ = pyminiply.read(
        location,
        read_normals=False,
        read_uv=False,
    )
    points = torch.from_numpy(points).T
    points = datasets.normalized(datasets.downsampled(points, num_points))

    return points


@torch.inference_mode()
def encode_and_decode(point_cloud: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    return model.forward(point_cloud.unsqueeze(0)).squeeze(0)


def show_dataset(
    server: viser.ViserServer,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    label: str,
    starting_position: numpy.ndarray,
    reconstruction_shift: numpy.ndarray,
    example_shift: numpy.ndarray,
) -> None:
    position = starting_position.copy()

    for i, point_cloud in enumerate(iter(dataset)):
        assert isinstance(point_cloud, torch.Tensor)

        point_cloud_reconstructed = encode_and_decode(point_cloud, model)
        point_cloud = point_cloud.T.numpy()
        point_cloud_reconstructed = point_cloud_reconstructed.T.numpy()

        colors = numpy.zeros_like(point_cloud)
        colors[:, 1] = 0.5

        colors_reconstructed = numpy.zeros_like(point_cloud)
        colors_reconstructed[:, 2] = 0.5

        server.scene.add_point_cloud(
            f"{label}_{i}",
            points=point_cloud,
            colors=colors,
            point_size=0.01,
            point_shape="circle",
            position=position,
        )
        server.scene.add_point_cloud(
            f"{label}_{i}_reconstructed",
            points=point_cloud_reconstructed,
            colors=colors_reconstructed,
            point_size=0.01,
            point_shape="circle",
            position=position + reconstruction_shift,
        )

        position += example_shift


def show(model_checkpoint: pathlib.Path, dataset_name: str, max_examples: int | None) -> None:
    server = viser.ViserServer()

    model = torch.load(model_checkpoint, weights_only=False, map_location=torch.device("cpu")).eval()
    logger.info("Model loaded")

    num_points = model.num_points
    assert isinstance(num_points, int)

    match dataset_name:
        case "ycb":
            dataset = datasets.YCBDataset(pathlib.Path("data/ycb"), num_points)

        case "geometric_shapes":
            dataset = datasets.GeometricShapesDataset(pathlib.Path("data/geometric_shapes"), num_points)

        case other:
            raise ValueError(f"Dataset `{other}` does not exist")

    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

    position_train = numpy.array((8.0, 0.0, 0.0))
    position_valid = numpy.array((0.0, 0.0, 0.0))
    position_test = numpy.array((-8.0, 0.0, 0.0))

    reconstruction_shift = numpy.array((-2.0, 0.0, 0.0))
    example_shift = numpy.array((0.0, -2.0, 0.0))

    show_dataset(server, model, dataset_train, "train", position_train, reconstruction_shift, example_shift)
    show_dataset(server, model, dataset_valid, "valid", position_valid, reconstruction_shift, example_shift)
    show_dataset(server, model, dataset_test, "test", position_test, reconstruction_shift, example_shift)

    server.sleep_forever()


if __name__ == "__main__":
    typer.run(show)
