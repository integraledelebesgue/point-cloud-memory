import pathlib

import numpy
import torch
import typer
import viser

import datasets

torch.manual_seed(0)


def show_dataset(
    server: viser.ViserServer,
    dataset: torch.utils.data.Dataset,
    label: str,
    starting_position: numpy.ndarray,
    example_shift: numpy.ndarray,
    max_examples: int,
) -> None:
    position = starting_position.copy()

    for i, bag in enumerate(iter(dataset)):
        if i == max_examples:
            break

        point_cloud_orginal = bag["point_cloud_original"]
        assert isinstance(point_cloud_orginal, torch.Tensor)

        point_clouds_masked = bag["point_clouds_masked"][1:]
        assert isinstance(point_clouds_masked, torch.Tensor)

        point_cloud_orginal = point_cloud_orginal.T.numpy()

        colors = numpy.zeros_like(point_cloud_orginal)
        colors[:, 1] = 0.5

        colors_masked = numpy.zeros_like(point_cloud_orginal)

        server.scene.add_point_cloud(
            f"{label}_{i}",
            points=point_cloud_orginal,
            colors=colors,
            point_size=0.01,
            point_shape="circle",
            position=position,
        )

        position += example_shift

        for j, point_cloud_masked in enumerate(point_clouds_masked.unbind(0)):
            server.scene.add_point_cloud(
                f"{label}_{i}_masked_{j}",
                points=point_cloud_masked.T.numpy(),
                colors=colors_masked,
                point_size=0.01,
                point_shape="circle",
                position=position,
            )

            position += example_shift


def show(dataset_name: str, max_examples: int, num_points: int, bag_size: int, drop_ratio: float) -> None:
    server = viser.ViserServer()

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

    position_train = numpy.array((4.0, 0.0, 0.0))
    position_valid = numpy.array((0.0, 0.0, 0.0))
    position_test = numpy.array((-4.0, 0.0, 0.0))
    example_shift = numpy.array((0.0, -2.0, 0.0))

    show_dataset(server, dataset_train, "train", position_train, example_shift, max_examples)
    show_dataset(server, dataset_valid, "valid", position_valid, example_shift, max_examples)
    show_dataset(server, dataset_test, "test", position_test, example_shift, max_examples)

    server.sleep_forever()


if __name__ == "__main__":
    typer.run(show)
