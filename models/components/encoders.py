import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import scatter


class SamplingEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.latent_dimension = encoder.latent_dimension // 2
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters = self.encoder.forward(x)

        mean = parameters[..., : self.latent_dimension]
        std = parameters[..., self.latent_dimension :]

        white_noise = torch.randn_like(mean, device=x.device)
        latent_sample = mean + white_noise * std
        return latent_sample


class PointNetBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.conv = torch.nn.Conv1d(input_channels, output_channels, 1)
        self.batch_norm = torch.nn.BatchNorm1d(output_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv.forward(x)
        x = self.batch_norm.forward(x)
        x = self.relu.forward(x)
        return x


class PointNetEncoder(torch.nn.Module):
    """
    Original PointNet (https://arxiv.org/pdf/1612.00593) without the T-Net modules.
    """

    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.block1 = PointNetBlock(3, 64)
        self.block2 = PointNetBlock(64, 128)
        self.block3 = PointNetBlock(128, latent_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 3  # (batch, 3, num_points)

        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)  # (batch, latent_dimension, num_points)
        x = x.amax(dim=-1)  # (batch, latent_dimension)

        return x


class PointNetWithHeadEncoder(torch.nn.Module):
    """
    Original PointNet (https://arxiv.org/pdf/1612.00593) without the T-Net modules.
    """

    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.block1 = PointNetBlock(3, 64)
        self.block2 = PointNetBlock(64, 128)
        self.block3 = PointNetBlock(128, latent_dimension)
        self.linear = torch.nn.Linear(latent_dimension, latent_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 3  # (batch, 3, num_points)

        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)  # (batch, latent_dimension, num_points)

        x = x.amax(dim=-1)  # (batch, latent_dimension)

        x = self.linear.forward(x)

        return x


class BigPointNetEncoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.block1 = PointNetBlock(3, 64)
        self.block2 = PointNetBlock(64, 128)
        self.block3 = PointNetBlock(128, 128)
        self.block4 = PointNetBlock(128, 256)
        self.block5 = PointNetBlock(256, latent_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 3  # (batch, 3, num_points)

        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.block4.forward(x)
        x = self.block5.forward(x)  # (batch, latent_dimension, num_points)
        x = x.amax(dim=-1)  # (batch, latent_dimension)

        return x


class BigPointNetWithHeadEncoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.conv1 = torch.nn.Sequential(torch.nn.Conv1d(3, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(torch.nn.Conv1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(torch.nn.Conv1d(64, 128, 1), torch.nn.BatchNorm1d(128), torch.nn.ReLU())
        self.conv5 = torch.nn.Sequential(torch.nn.Conv1d(128, 1024, 1), torch.nn.BatchNorm1d(1024), torch.nn.ReLU())

        self.max_pool = torch.nn.MaxPool1d(num_points, return_indices=False)

        self.fc1 = torch.nn.Sequential(torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(256, latent_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 3  # (batch, 3, num_points)

        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        x = self.conv4.forward(x)
        x = self.conv5.forward(x)  # (batch, 1024, num_points)

        x = self.max_pool.forward(x)  # (batch, 1024)
        x = x.squeeze(-1)

        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        x = self.dropout.forward(x)
        x = self.fc3.forward(x)

        return x


class LocalCovariance(torch.nn.Module):
    """
    Computes a 3x3 local covariance matrix for the k-neighborhood of each point.
    Input: (batch, 3)
    Output: (batch, 12) (3 coordinates + 9 vectorized covariance values)
    """

    def __init__(self, k: int = 16) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Compute K-nearest neighbors graph
        edge_index = knn_graph(x, k=self.k, batch=batch)
        row, col = edge_index.unbind(0)

        neighbors = x[col].view(-1, self.k, 3)  # (batch_size, k, 3)
        mean = neighbors.mean(dim=1, keepdim=True)
        neighbors_centered = neighbors - mean

        covariance = neighbors_centered.transpose(1, 2) @ neighbors_centered / self.k  # (batch_size, 3, 3)
        covariance = covariance.reshape(-1, 9)  # (batch_size, 9)

        return torch.cat([x, covariance], dim=1)  # (batch_size, 12)


class GraphLayer(torch.nn.Module):
    """
    FoldingNet Graph Layer: Max-pooling over local neighborhood.
    """

    def __init__(self, input_channels: int, output_channels: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.linear = torch.nn.Linear(input_channels, output_channels)
        self.batch_norm = torch.nn.BatchNorm1d(output_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        edge_index = knn_graph(x, k=self.k, batch=batch)
        row, col = edge_index.unbind(0)

        # Local Max-Pooling: aggregation from neighbors (col) to central points (row)
        out = scatter(x[col], row, dim=0, reduce="max")

        out = self.linear.forward(out)
        out = self.batch_norm.forward(out)
        out = self.relu.forward(out)

        return out


class FoldingNetEncoder(torch.nn.Module):
    """
    Encoder: Extracts a 512-dim codeword from a point cloud.
    """

    def __init__(self, num_points: int, latent_dimension: int, k: int = 16) -> None:
        super().__init__()

        self.latent_dimension = latent_dimension
        self.num_points = num_points

        self.local_covariance = LocalCovariance(k=k)

        # 3-layer point-wise MLP: {64, 64, 64}
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(12, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.graph1 = GraphLayer(64, 128, k=k)
        self.graph2 = GraphLayer(128, 512, k=k)

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, latent_dimension),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.local_covariance.forward(x, batch)
        x = self.mlp.forward(x)
        x = self.graph1.forward(x, batch)
        x = self.graph2.forward(x, batch)

        x = scatter(x, batch, dim=0, reduce="max")

        x = self.bottleneck.forward(x)

        return x
