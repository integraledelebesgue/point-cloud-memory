import pathlib

import numpy
import typer
import viser

import datasets


def show(dataset_name: str, num_points: int) -> None:
    server = viser.ViserServer()

    match dataset_name:
        case "ycb":
            dataset = datasets.YCBDataset(
                pathlib.Path("data/ycb"),
                num_points,
                apply_transformations=False,
            )
            dataset_transformed = datasets.YCBDataset(
                pathlib.Path("data/ycb"),
                num_points,
                apply_transformations=True,
            )

        case "geometric_shapes":
            dataset = datasets.GeometricShapesDataset(
                pathlib.Path("data/geometric_shapes"),
                num_points,
                apply_transformations=False,
            )
            dataset_transformed = datasets.GeometricShapesDataset(
                pathlib.Path("data/geometric_shapes"),
                num_points,
                apply_transformations=True,
            )

        case other:
            raise ValueError(f"Dataset `{other}` does not exist")

    position = numpy.array((0.0, 0.0, 0.0))

    num_rows = 10
    num_examples = len(dataset)

    for i in range(num_examples):
        row = i // num_rows
        col = i % num_rows

        point_cloud = dataset[i].T.numpy()
        point_cloud_normalized = dataset_transformed[i].T.numpy()

        colors = numpy.zeros_like(point_cloud)
        colors[:, 1] = 0.5

        colors_normalized = numpy.zeros_like(point_cloud)
        colors_normalized[:, 2] = 0.5

        position = numpy.array((row * 4.0, col * 4.0, 0.0))
        position_normalized = numpy.array((row * 4.0 + 2.0, col * 4.0, 0.0))

        server.scene.add_point_cloud(
            f"example_{i}",
            points=point_cloud,
            colors=colors,
            point_size=0.01,
            point_shape="circle",
            position=position,
        )
        server.scene.add_point_cloud(
            f"example_{i}_normalized",
            points=point_cloud_normalized,
            colors=colors_normalized,
            point_size=0.01,
            point_shape="circle",
            position=position_normalized,
        )

    server.sleep_forever()


if __name__ == "__main__":
    typer.run(show)
