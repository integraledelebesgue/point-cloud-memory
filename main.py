import pathlib

import torch
import typer
from loguru import logger

import training
from dataset import YCBDataset
from network import PointCloudVAE

NUM_POINTS = 100_000
LATENT_DIMENSION = 32

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3


def train(num_epochs: int, learning_rate: float, name: str, torch_device: str = "mps") -> None:
    dataset_root = pathlib.Path(__file__).parent / "data" / "ycb"
    dataset = YCBDataset(dataset_root, num_points=NUM_POINTS)

    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=BATCH_SIZE)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

    device = torch.device(torch_device)
    model = PointCloudVAE(num_points=NUM_POINTS, latent_dimension=LATENT_DIMENSION).to(device)

    logger.info(
        f"Loaded {len(dataset)} examples (training: {len(dataset_train)}, validation {len(dataset_valid)}, test: {len(dataset_test)})"
    )

    training.train(model, data_loader_train, data_loader_valid, num_epochs, learning_rate, device)

    destination = (pathlib.Path(__file__).parent / "models" / name).with_suffix(".pt")
    if destination.exists():
        logger.warning(f"Overwriting the existing model at {destination}")

    torch.save(model, destination)

    test_loss_avg = training.evaluate(model, data_loader_test, device)
    print(f"Test loss: {test_loss_avg:.2f}")


if __name__ == "__main__":
    typer.run(train)
