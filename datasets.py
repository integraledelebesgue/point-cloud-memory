import pathlib

import pyminiply
import torch
import torch_geometric.datasets as pyg_datasets
import torch_geometric.transforms as pyg_transforms


def read_point_cloud(source: pathlib.Path) -> torch.Tensor:
    points, *_ = pyminiply.read(
        source,
        read_normals=False,
        read_uv=False,
        read_color=False,
    )

    points = torch.from_numpy(points).to(torch.float32).T  # (3, num_points)
    # colors = torch.from_numpy(colors).to(torch.float32)

    return points


def downsampled(point_cloud: torch.Tensor, num_points: int) -> torch.Tensor:
    num_channels, num_points_in_cloud = point_cloud.shape
    num_points_to_copy = min(num_points, num_points_in_cloud)

    indices_to_copy = torch.randperm(num_points_in_cloud)[:num_points_to_copy]

    new_point_cloud = torch.zeros(num_channels, num_points)
    new_point_cloud[..., :num_points_to_copy] = point_cloud[..., indices_to_copy]

    return new_point_cloud


def normalized(point_cloud: torch.Tensor) -> torch.Tensor:
    mean = point_cloud.mean(dim=1).reshape(3, 1)
    point_cloud_centered = point_cloud - mean
    radius = point_cloud_centered.norm(dim=0).max()
    point_cloud_normalized = point_cloud_centered / radius
    return point_cloud_normalized


class YCBDataset(torch.utils.data.Dataset):
    point_clouds: list[torch.Tensor]

    def __init__(self, root: pathlib.Path, num_points: int, apply_transformations: bool = True) -> None:
        point_cloud_files = list(root.glob("*/clouds/merged_cloud.ply"))
        point_cloud_files.sort(key=lambda file: file.parent.parent.name)

        point_clouds = [downsampled(read_point_cloud(source), num_points) for source in point_cloud_files]

        if apply_transformations:
            point_clouds = [normalized(point_cloud) for point_cloud in point_clouds]

        self.point_clouds = point_clouds

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.point_clouds[index]

    def __len__(self) -> int:
        return len(self.point_clouds)


class GeometricShapesDataset(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path, num_points: int, apply_transformations: bool = True) -> None:
        self.apply_transformations = apply_transformations
        self.dataset = pyg_datasets.GeometricShapes(
            str(root),
            transform=pyg_transforms.SamplePoints(num_points),
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.dataset[index]
        assert hasattr(data, "pos")

        point_cloud = data.pos
        assert isinstance(point_cloud, torch.Tensor)

        point_cloud = point_cloud.T  # (num_points, 3) -> (3, num_points)

        if self.apply_transformations:
            point_cloud = normalized(point_cloud)

        return point_cloud

    def __len__(self) -> int:
        return len(self.dataset)
