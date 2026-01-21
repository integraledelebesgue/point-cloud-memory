import pathlib

import torch
import typer
from loguru import logger

import datasets
import models
import training
from models.components import encoders

NUM_POINTS = 2048
LATENT_DIMENSION = 1024
BAG_SIZE = 32

torch.manual_seed(0)


def main(num_epochs: int, learning_rate: float, batch_size: int, name: str, torch_device: str) -> None:
    data_directory = pathlib.Path(__file__).parent / "data"

    dataset_base = datasets.YCBDataset(data_directory / "ycb", num_points=NUM_POINTS)
    dataset_base_train, dataset_base_valid, dataset_base_test = torch.utils.data.random_split(dataset_base, [0.7, 0.15, 0.15])
    dataset_train = datasets.MaskedBagsDataset(dataset_base_train, num_points=NUM_POINTS, bag_size=BAG_SIZE, drop_ratio=0.8)
    dataset_valid = datasets.MaskedBagsDataset(dataset_base_valid, num_points=NUM_POINTS, bag_size=BAG_SIZE, drop_ratio=0.8)
    dataset_test = datasets.MaskedBagsDataset(dataset_base_test, num_points=NUM_POINTS, bag_size=BAG_SIZE, drop_ratio=0.8)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    logger.info(f"Training: {len(dataset_train)}, validation {len(dataset_valid)}, test: {len(dataset_test)})")

    device = torch.device(torch_device)

    vae = torch.load("checkpoints/vae_point_net_small_&_linear_emd_adamw_200.pt", weights_only=False, map_location="cpu").to(device)
    encoder = encoders.SamplingEncoderWrapper(vae.encoder)
    decoder = vae.decoder

    for param in encoder.parameters():
        param.requires_grad = False

    memory = models.memory.PointCloudMemory(
        encoder,
        decoder,
        hidden_size=256,
        quantity=len(dataset_base_train),
    ).to(device)

    # model = models.ae.VariationalAutoencoder(
    #     encoders.PointNetEncoder(latent_dimension=2 * LATENT_DIMENSION, num_points=NUM_POINTS),
    #     decoders.LinearDecoder(latent_dimension=LATENT_DIMENSION, num_points=NUM_POINTS),
    # ).to(device)

    optimizer = torch.optim.AdamW(memory.memory_parameters(), lr=learning_rate, fused=True)
    criterion = training.metrics.EarthMoversDistance(epsilon=1e-1, max_iterations=100)

    training.hopfield.train(
        encoder,
        decoder,
        memory,
        optimizer,
        criterion,
        data_loader_train,
        data_loader_valid,
        num_epochs,
        learning_rate,
        device,
    )

    destination = (pathlib.Path(__file__).parent / "checkpoints" / name).with_suffix(".pt")
    if destination.exists():
        logger.warning(f"Overwriting the existing model at {destination}")

    torch.save(memory.cpu(), destination)

    # test_loss_avg = training.evaluate(model, criterion, data_loader_test, device)
    # print(f"Test loss: {test_loss_avg:.2f}")


if __name__ == "__main__":
    typer.run(main)
