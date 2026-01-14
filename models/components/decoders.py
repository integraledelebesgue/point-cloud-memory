import math

import torch


class LinearDecoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(latent_dimension, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.linear3 = torch.nn.Linear(256, 3 * num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)
        x = self.linear3.forward(x)
        x = x.reshape(-1, 3, self.num_points)
        return x


def get_grid(num_points: int) -> torch.Tensor:
    grid_size = int(math.ceil(math.sqrt(num_points)))

    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    grid = torch.stack((xx, yy), dim=0)  # (2, grid_size, grid_size)
    grid = grid.flatten(1, 2)  # (2, grid_size * grid_size)
    grid = grid[..., :num_points]  # if grid_size * grid_size > num_points

    return grid


class FoldingLinearDecoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.grid = get_grid(num_points)

        self.conv1 = torch.nn.Conv1d(latent_dimension + 2, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _num_points = x.shape
        grid = self.grid.repeat(batch_size, 1, 1).to(x.device)  # (batch_size, 2, num_points)

        # Concatenate the latent vector to every point in the grid:
        latent_repeated = x.unsqueeze(2).repeat(1, 1, self.grid.shape[1])  # (batch_size, latent_dimension, num_points)
        x = torch.cat([grid, latent_repeated], dim=1)  # (batch_size, 2 + latent_dimension, num_points)

        x = torch.nn.functional.relu(self.conv1.forward(x))  # (batch_size, 512, num_points)
        x = torch.nn.functional.relu(self.conv2.forward(x))  # (batch_size, 512, num_points)
        x = self.conv3.forward(x)  # (batch_size, 3, num_points)

        return x


class DoubleFoldingLinearDecoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.grid = get_grid(num_points)

        self.folding1 = torch.nn.Conv1d(latent_dimension + 2, 3, 1)
        self.folding2 = torch.nn.Conv1d(latent_dimension + 3, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _num_points = x.shape
        grid = self.grid.to(x.device).repeat(batch_size, 1, 1)  # (batch_size, 2, num_points)

        # Concatenate the latent vector to every point in the grid:
        latent_repeated = x.unsqueeze(2).repeat(1, 1, self.grid.shape[1])  # (batch_size, latent_dimension, num_points)

        # First folding:
        x = torch.cat([grid, latent_repeated], dim=1)  # (batch_size, 2 + latent_dimension, num_points)
        x = self.folding1(x)  # (batch_size, 3, num_points)

        # Second folding:
        x = torch.cat([x, latent_repeated], dim=1)  # (batch_size, 3 + latent_dimension, num_points)
        x = self.folding2(x)  # (batch_size, 3, num_points)

        return x


class LinearBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        self.linear = torch.nn.Linear(input_channels, output_channels)
        self.batch_norm = torch.nn.BatchNorm1d(output_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear.forward(x)
        x = self.batch_norm.forward(x)
        x = self.relu.forward(x)
        return x


class UpConvBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        self.deconv = torch.nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.batch_norm = torch.nn.BatchNorm2d(output_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv.forward(x)
        x = self.batch_norm.forward(x)
        x = self.relu.forward(x)
        return x


class UpConvDecoder(torch.nn.Module):
    def __init__(self, latent_dimension: int, num_points: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        # Initial projection to a 4x4 spatial grid
        self.projection1 = LinearBlock(latent_dimension, 1024)
        self.projection2 = LinearBlock(1024, 256 * 4 * 4)

        # Transposed convolutions to upsample the grid:
        self.block1 = UpConvBlock(256, 128)
        self.block2 = UpConvBlock(128, 64)
        self.block3 = UpConvBlock(64, 32)

        self.deconv = torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.projection1.forward(x)
        x = self.projection2.forward(x)

        x = x.reshape(-1, 256, 4, 4)  # reshape to spatial grid

        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.deconv.forward(x)  # (batch_size, 3, 64, 64)

        x = x.flatten(-2, -1)  # (batch_size, 3, 64 * 64)

        return x[..., : self.num_points]


class ReversedPointNetWithHeadDecoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.fc3_transpose = torch.nn.Linear(latent_dimension, 128)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2_transpose = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.fc1_transpose = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())

        self.unpool = torch.nn.Linear(256, 256 * num_points)

        self.conv5_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(256, 128, 1), torch.nn.BatchNorm1d(128), torch.nn.ReLU())
        self.conv4_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(128, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv3_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv2_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv1_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(64, 3, 1), torch.nn.BatchNorm1d(3), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc3_transpose.forward(x)
        x = self.dropout.forward(x)
        x = self.fc2_transpose.forward(x)
        x = self.fc1_transpose.forward(x)

        x = self.unpool.forward(x)
        x = x.reshape(-1, 256, self.num_points)

        x = self.conv5_transpose.forward(x)
        x = self.conv4_transpose.forward(x)
        x = self.conv3_transpose.forward(x)
        x = self.conv2_transpose.forward(x)
        x = self.conv1_transpose.forward(x)

        return x


class FoldingNetDecoder(torch.nn.Module):
    """
    Decoder: Deforms a 2D grid into a 3D surface based on the codeword.
    """

    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        # Folding Stage 1: codeword (512) + grid (2) -> 3D
        self.folding1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension + 2, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 3),
        )

        # Folding Stage 2: codeword (512) + result of fold 1 (3) -> 3D
        self.folding2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension + 3, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 3),
        )

        self.grid = get_grid(num_points).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_points = self.num_points
        batch_size, _num_points = x.shape

        grid = self.grid.repeat(batch_size, 1, 1).to(x.device)  # (batch_size, 2, num_points)

        # Concatenate the latent vector to every point in the grid:
        latent_repeated = x.unsqueeze(2).repeat(1, 1, self.grid.shape[1])  # (batch_size, latent_dimension, num_points)

        # First folding:
        x = torch.cat((latent_repeated, grid), dim=-1).reshape(-1, self.latent_dimension + 2)
        x = self.folding1.forward(x).reshape(batch_size, num_points, 3)

        # Second folding:
        x = torch.cat((x, grid), dim=-1).reshape(-1, self.latent_dimension + 3)
        x = self.folding2.forward(x).reshape(batch_size, num_points)

        return x
