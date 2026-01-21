import torch
import tqdm


@torch.no_grad()
def encode_point_cloud_bag(bag: torch.Tensor, encoder: torch.nn.Module) -> torch.Tensor:
    batch_size, bag_size, *_ = bag.shape
    bag_batch_flattened = bag.flatten(0, 1)
    bag_batch_flattened_encoded: torch.Tensor = encoder.forward(bag_batch_flattened)
    bag_batch_encoded = bag_batch_flattened_encoded.unflatten(0, (batch_size, bag_size))
    return bag_batch_encoded


def train(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    memory: torch.nn.Module,
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

        for bag in tqdm.tqdm(data_loader_train, desc=f"Training [{epoch + 1}/{num_epochs}]"):
            point_cloud_original = bag["point_cloud_original"].to(device)
            point_clouds_masked = bag["point_clouds_masked"].to(device)

            point_clouds_reconstructed = memory.forward(point_clouds_masked.flatten(0, 1))

            # point_clouds_masked_encoded = encode_point_cloud_bag(point_clouds_masked, encoder)
            # point_clouds_reconstructed_encoded = memory.forward(point_clouds_masked_encoded).flatten(0, 1)
            # point_clouds_reconstructed = decoder.forward(point_clouds_reconstructed_encoded)

            bag_size = point_clouds_masked.size(1)
            point_cloud_original_repeated = point_cloud_original.repeat(bag_size, 1, 1)

            # print(f"{point_clouds_reconstructed.shape = }")
            # print(f"{point_cloud_original.shape = }")
            # print(f"{point_cloud_original_repeated.shape = }")

            loss = criterion.forward(point_cloud_original_repeated, point_clouds_reconstructed)
            loss.backward()
            epoch_loss_total += loss.mean().detach().cpu().item()

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss_avg = epoch_loss_total / len(data_loader_train)
        # validation_loss_avg = evaluate(model, criterion, data_loader_valid, device)

        print(f"Average loss: training: {epoch_loss_avg:.2f}")
        # print(f"Average loss: training: {epoch_loss_avg:.2f}, validation: {validation_loss_avg:.2f}")
