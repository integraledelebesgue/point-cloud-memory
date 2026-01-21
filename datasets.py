import pathlib
import typing

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


def mask_random_patch(point_cloud: torch.Tensor, num_neighbors: int) -> torch.Tensor:
    num_points = point_cloud.size(1)

    seed_index = torch.randint(0, num_points, (1,))
    seed = point_cloud[:, seed_index]

    distance = (point_cloud - seed).pow(2).sum(dim=0)
    _, patch_indices = torch.topk(distance, num_neighbors, largest=False)

    masked_cloud = point_cloud.clone()
    masked_cloud[:, patch_indices] = point_cloud[:, 0].unsqueeze(1)

    return masked_cloud


def generate_bag(point_cloud: torch.Tensor, bag_size: int, drop_ratio: float) -> torch.Tensor:
    num_points = point_cloud.size(1)
    patch_size = int(num_points * drop_ratio)
    masked_clouds = [point_cloud]

    for _ in range(bag_size - 1):
        point_cloud_masked = mask_random_patch(point_cloud, patch_size)
        masked_clouds.append(point_cloud_masked)

    return torch.stack(masked_clouds)


class Bag(typing.TypedDict):
    point_cloud_original: torch.Tensor
    point_clouds_masked: torch.Tensor


class MaskedBagsDataset(torch.utils.data.Dataset):
    point_cloud_bags: list[Bag]

    def __init__(self, base_dataset: torch.utils.data.Dataset, num_points: int, bag_size: int, drop_ratio: float) -> None:
        self.point_cloud_bags = [
            {"point_cloud_original": point_cloud, "point_clouds_masked": generate_bag(point_cloud, bag_size, drop_ratio)}
            for point_cloud in iter(base_dataset)
        ]

    def __getitem__(self, index: int) -> Bag:
        return self.point_cloud_bags[index]

    def __len__(self) -> int:
        return len(self.point_cloud_bags)


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
