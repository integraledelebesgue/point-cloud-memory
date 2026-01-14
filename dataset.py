import pathlib

import pyminiply
import torch


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


def to_size(point_cloud: torch.Tensor, num_points: int) -> torch.Tensor:
    num_channels, num_points_in_cloud = point_cloud.shape
    num_points_to_copy = min(num_points, num_points_in_cloud)

    indices_to_copy = torch.randperm(num_points_in_cloud)[:num_points_to_copy]

    new_point_cloud = torch.zeros(num_channels, num_points)
    new_point_cloud[..., :num_points_to_copy] = point_cloud[..., indices_to_copy]

    return new_point_cloud


def normalize(point_cloud: torch.Tensor) -> torch.Tensor:
    min = point_cloud.min(dim=1, keepdim=True).values
    max = point_cloud.max(dim=1, keepdim=True).values
    normalized = (point_cloud - min) / (max - min)
    return normalized


class YCBDataset(torch.utils.data.Dataset):
    point_clouds: list[torch.Tensor]

    def __init__(self, root: pathlib.Path, num_points: int) -> None:
        point_cloud_files = root.glob("*/clouds/merged_cloud.ply")
        self.point_clouds = [normalize(to_size(read_point_cloud(source), num_points)) for source in point_cloud_files]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.point_clouds[index]

    def __len__(self) -> int:
        return len(self.point_clouds)
