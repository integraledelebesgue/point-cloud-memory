import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        assert encoder.num_points == decoder.num_points
        assert encoder.latent_dimension == decoder.latent_dimension

        self.num_points: int = encoder.num_points
        self.latent_dimension: int = encoder.latent_dimension

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder.forward(x)
        out = self.decoder.forward(out)
        return out


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()

        assert isinstance(encoder.num_points, int)
        assert isinstance(decoder.num_points, int)
        assert isinstance(encoder.latent_dimension, int)
        assert isinstance(decoder.latent_dimension, int)

        assert encoder.num_points == decoder.num_points
        assert encoder.latent_dimension == 2 * decoder.latent_dimension

        self.num_points: int = decoder.num_points
        self.latent_dimension: int = decoder.latent_dimension

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters = self.encoder.forward(x)

        mean = parameters[..., : self.latent_dimension]
        std = parameters[..., self.latent_dimension :]

        white_noise = torch.randn_like(mean, device=x.device)
        latent_sample = mean + white_noise * std

        out = self.decoder.forward(latent_sample)
        return out
