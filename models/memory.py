import torch
from hflayers import HopfieldLayer


class PointCloudMemory(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        hidden_size: int,
        quantity: int,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.memory = HopfieldLayer(
            input_size=decoder.latent_dimension,
            hidden_size=hidden_size,
            output_size=decoder.latent_dimension,
            quantity=quantity,
        )

    def memory_parameters(self):
        return self.memory.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward(x)
        x = x.unsqueeze(1)
        x = self.memory.forward(x)
        x = x.squeeze(1)
        x = self.decoder.forward(x)
        return x
