import pathlib

import numpy
import torch
import typer
import viser

import datasets

torch.manual_seed(0)


def show_dataset(
    server: viser.ViserServer,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    label: str,
    starting_position: numpy.ndarray,
    reconstruction_shift: numpy.ndarray,
    example_shift: numpy.ndarray,
    max_examples: int,
) -> None:
    position = starting_position.copy()

    for i, bag in enumerate(iter(dataset)):
        if i == max_examples:
            break

        point_cloud_orginal = bag["point_cloud_original"]
        point_cloud_orginal = point_cloud_orginal.T.numpy()

        point_cloud_masked = bag["point_clouds_masked"][1, ...]
        with torch.inference_mode():
            point_cloud_reconstructed = model.forward(point_cloud_masked.unsqueeze(0)).squeeze(0).T.numpy()

        colors = numpy.zeros_like(point_cloud_orginal)
        colors[:, 1] = 0.5

        colors_reconstructed = numpy.zeros_like(point_cloud_orginal)
        colors_reconstructed[:, 2] = 0.5

        server.scene.add_point_cloud(
            f"{label}_{i}",
            points=point_cloud_orginal,
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


def show(model_checkpoint: pathlib.Path, dataset_name: str, max_examples: int, bag_size: int, drop_ratio: float) -> None:
    server = viser.ViserServer()

    model = torch.load(model_checkpoint, weights_only=False, map_location="cpu").eval()
    num_points = model.decoder.num_points

    match dataset_name:
        case "ycb":
            base_dataset = datasets.YCBDataset(pathlib.Path("data/ycb"), num_points)

        case "geometric_shapes":
            base_dataset = datasets.GeometricShapesDataset(pathlib.Path("data/geometric_shapes"), num_points)

        case other:
            raise ValueError(f"Dataset `{other}` does not exist")

    base_dataset_train, base_dataset_valid, base_dataset_test = torch.utils.data.random_split(base_dataset, [0.7, 0.15, 0.15])

    dataset_train = datasets.MaskedBagsDataset(base_dataset_train, num_points, bag_size, drop_ratio)
    dataset_valid = datasets.MaskedBagsDataset(base_dataset_valid, num_points, bag_size, drop_ratio)
    dataset_test = datasets.MaskedBagsDataset(base_dataset_test, num_points, bag_size, drop_ratio)

    position_train = numpy.array((8.0, 0.0, 0.0))
    position_valid = numpy.array((0.0, 0.0, 0.0))
    position_test = numpy.array((-8.0, 0.0, 0.0))

    reconstruction_shift = numpy.array((-2.0, 0.0, 0.0))
    example_shift = numpy.array((0.0, -2.0, 0.0))

    show_dataset(server, model, dataset_train, "train", position_train, reconstruction_shift, example_shift, max_examples)
    show_dataset(server, model, dataset_valid, "valid", position_valid, reconstruction_shift, example_shift, max_examples)
    show_dataset(server, model, dataset_test, "test", position_test, reconstruction_shift, example_shift, max_examples)

    server.sleep_forever()


if __name__ == "__main__":
    typer.run(show)
