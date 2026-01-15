import hflayers
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(PointNetEncoder, self).__init__()
        # Shared MLP: [64, 128, 128, 256, 1024]
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, latent_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        # Input x shape: (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global Max Pooling
        x = x.amax(dim=2)
        return x


class UpConvDecoder(nn.Module):
    def __init__(self, latent_dim=1024, num_points=2048):
        super(UpConvDecoder, self).__init__()
        self.num_points = num_points

        # Initial projection to a 4x4 spatial grid
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 256 * 4 * 4)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(256 * 4 * 4)

        # Transposed convolutions to upsample the grid
        # (B, 256, 4, 4) -> (B, 128, 8, 8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # (B, 128, 8, 8) -> (B, 64, 16, 16)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # (B, 64, 16, 16) -> (B, 32, 32, 32)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # (B, 32, 32, 32) -> (B, 3, 64, 64)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, z):
        # (B, 1024) -> (B, 4096)
        x = F.relu(self.bn_fc1(self.fc1(z)))
        x = F.relu(self.bn_fc2(self.fc2(x)))

        # Reshape to spatial grid: (Batch, Channels, Height, Width)
        x = x.view(-1, 256, 4, 4)

        # Upsampling layers
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))

        # Final layer (usually no activation or Tanh depending on data normalization)
        x = self.deconv4(x)  # Shape: (B, 3, 64, 64)

        # Reshape to (B, 3, 4096) and take the first num_points
        x = x.view(-1, 3, 64 * 64)
        return x[:, :, : self.num_points]


class PointNetAE(nn.Module):
    def __init__(self, latent_dim, num_points=2048):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim=latent_dim)
        self.decoder = UpConvDecoder(latent_dim=latent_dim, num_points=num_points)

    def forward(self, x):
        # x: (B, 3, N)
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction


class PointCloudVAE(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(num_points, 2 * latent_dimension)
        self.decoder = Decoder(num_points, latent_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters, max_pool_indices = self.encoder.forward(x)

        num_points = x.shape[2]
        assert num_points == self.num_points

        match x.shape:
            case 3, _:
                mean = parameters[: self.latent_dimension]
                std_dev = parameters[self.latent_dimension :]
                white_noise = torch.randn(self.latent_dimension, device=x.device)

            case batch_size, 3, _:
                mean = parameters[:, : self.latent_dimension]
                std_dev = parameters[:, self.latent_dimension :]
                white_noise = torch.randn(batch_size, self.latent_dimension, device=x.device)

            case _:
                raise

        embedding = mean + std_dev * white_noise

        return self.decoder.forward(embedding, max_pool_indices)


class Encoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.conv1 = torch.nn.Sequential(torch.nn.Conv1d(3, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(torch.nn.Conv1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(torch.nn.Conv1d(64, 128, 1), torch.nn.BatchNorm1d(128), torch.nn.ReLU())
        self.conv5 = torch.nn.Sequential(torch.nn.Conv1d(128, 1024, 1), torch.nn.BatchNorm1d(1024), torch.nn.ReLU())

        self.max_pool = torch.nn.MaxPool1d(num_points, return_indices=True)

        self.fc1 = torch.nn.Sequential(torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(256, latent_dimension)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == 3  # (batch, 3, num_points)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # (batch, 1024, num_points)

        x, indices = self.max_pool.forward(x)  # (batch, 1024)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x, indices


class Decoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.latent_dimension = latent_dimension

        self.fc3_transpose = torch.nn.Linear(latent_dimension, 256)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2_transpose = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU())
        self.fc1_transpose = torch.nn.Sequential(torch.nn.Linear(512, 1024), torch.nn.BatchNorm1d(1024), torch.nn.ReLU())

        self.max_unpool = torch.nn.MaxUnpool1d(num_points)

        self.conv5_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(1024, 128, 1), torch.nn.BatchNorm1d(128), torch.nn.ReLU())
        self.conv4_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(128, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv3_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv2_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(64, 64, 1), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
        self.conv1_transpose = torch.nn.Sequential(torch.nn.ConvTranspose1d(64, 3, 1), torch.nn.BatchNorm1d(3), torch.nn.ReLU())

    def forward(self, x: torch.Tensor, max_pool_indices: torch.Tensor) -> torch.Tensor:
        x = self.fc3_transpose.forward(x)
        x = self.dropout.forward(x)
        x = self.fc2_transpose.forward(x)
        x = self.fc1_transpose.forward(x)

        x = x.unsqueeze(-1)
        x = self.max_unpool.forward(x, max_pool_indices)

        x = self.conv5_transpose.forward(x)
        x = self.conv4_transpose.forward(x)
        x = self.conv3_transpose.forward(x)
        x = self.conv2_transpose.forward(x)
        x = self.conv1_transpose.forward(x)

        return x


class PointCloudMemory(torch.nn.Module):
    def __init__(self, num_points: int, latent_dimension: int) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dimension = latent_dimension

        self.encoder = Encoder(num_points, 2 * latent_dimension)
        self.hopfield_layer = hflayers.Hopfield(latent_dimension)
        self.decoder = Decoder(num_points, latent_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters, max_pool_indices = self.encoder.forward(x)

        match x.shape:
            case 3, _:
                mean = parameters[: self.latent_dimension]
                std_dev = parameters[self.latent_dimension :]
                white_noise = torch.randn(self.latent_dimension, device=x.device)

            case batch_size, 3, _:
                mean = parameters[:, : self.latent_dimension]
                std_dev = parameters[:, self.latent_dimension :]
                white_noise = torch.randn(batch_size, self.latent_dimension, device=x.device)

            case _:
                raise

        embedding = mean + std_dev * white_noise
        embedding_associated = self.hopfield_layer.forward(embedding.unsqueeze(-2)).squeeze(-2)
        out = self.decoder.forward(embedding_associated, max_pool_indices)

        return out
