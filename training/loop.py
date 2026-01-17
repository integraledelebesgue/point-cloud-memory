import torch
import tqdm


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    total_loss = 0.0

    for point_cloud in tqdm.tqdm(data_loader, desc="Evaluating"):
        point_cloud = point_cloud.to(device)
        point_cloud_reconstructed = model.forward(point_cloud)

        loss = criterion.forward(point_cloud, point_cloud_reconstructed)
        total_loss += loss.cpu().mean().item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_valid: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    for epoch in range(num_epochs):
        epoch_loss_total = 0.0

        for point_cloud in tqdm.tqdm(data_loader_train, desc=f"Training [{epoch + 1}/{num_epochs}]"):
            point_cloud = point_cloud.to(device)
            point_cloud_reconstructed = model.forward(point_cloud)

            loss = criterion.forward(point_cloud, point_cloud_reconstructed)
            loss.backward()
            epoch_loss_total += loss.mean().detach().cpu().item()

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss_avg = epoch_loss_total / len(data_loader_train)
        validation_loss_avg = evaluate(model, criterion, data_loader_valid, device)

        print(f"Average loss: training: {epoch_loss_avg:.2f}, validation: {validation_loss_avg:.2f}")
