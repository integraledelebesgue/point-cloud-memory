import typing

import torch
from pytorch3d.loss.chamfer import chamfer_distance


class ChamferDistance(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.device == y.device, f"x and y should be on the same device (x: {x.device}, y: {y.device})"
        device = y.device

        x = x.cpu()
        y = y.cpu()

        distance, _ = chamfer_distance(x, y)
        return distance.to(device)


class AugmentedChamferDistance(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.device == y.device, f"x and y should be on the same device (x: {x.device}, y: {y.device})"
        device = y.device

        x = x.cpu()
        y = y.cpu()

        x_to_y, _ = chamfer_distance(x, y, single_directional=True)
        y_to_x, _ = chamfer_distance(y, x, single_directional=True)

        return torch.maximum(x_to_y, y_to_x).to(device)


class EarthMoversDistance(torch.nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-1,
        max_iterations: int = 100,
    ) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        B, N, _ = x.shape
        C = torch.cdist(x, y, p=2) ** 2

        K = torch.exp(-C / self.epsilon)
        u = torch.ones(B, N, device=x.device) / N
        v = torch.ones(B, N, device=x.device) / N

        for _ in range(self.max_iterations):
            v = (1.0 / N) / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
            u = (1.0 / N) / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)

        plan = u.unsqueeze(2) * K * v.unsqueeze(1)
        return torch.sum(plan * C) / B


class SinkhornDistance(torch.nn.Module):
    """
    An approximate Earth Mover's Distance (EMD) implementation
    using the Sinkhorn algorithm.
    """

    def __init__(
        self,
        epsilon: float = 1e-1,
        max_iterations: int = 100,
        reduction: typing.Literal["sum", "mean"] = "mean",
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), y: (B, N, 3)
        # The algorithm expects (B, N, D)

        # Ensure points are in the right shape (B, N, 3)
        if x.shape[1] == 3 and x.shape[2] != 3:
            x = x.transpose(1, 2)
        if y.shape[1] == 3 and y.shape[2] != 3:
            y = y.transpose(1, 2)

        B, N, D = x.shape

        # Compute pairwise distance matrix (Cost matrix)
        # C_{ij} = ||x_i - y_j||^2
        C = torch.cdist(x, y, p=2) ** 2  # (B, N, N)

        # Initialize potentials
        mu = torch.empty(B, N, device=x.device).fill_(1.0 / N)
        nu = torch.empty(B, N, device=x.device).fill_(1.0 / N)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Sinkhorn iterations (in log space for stability)
        actual_niter = 0
        thresh = 1e-1

        for i in range(self.max_iterations):
            u_prev = u
            # Log-sum-exp update
            u = self.epsilon * (torch.log(mu + 1e-8) - torch.logsumexp(self._M(C, u, v) / self.epsilon, dim=-1)) + u
            v = self.epsilon * (torch.log(nu + 1e-8) - torch.logsumexp(self._M(C, u, v).transpose(-2, -1) / self.epsilon, dim=-1)) + v

            err = (u - u_prev).abs().mean()
            actual_niter += 1
            if err < thresh:
                break

        # Compute the transport plan PI
        # PI = exp((u + v - C) / eps)
        PI = torch.exp(self._M(C, u, v) / self.epsilon)

        # Sinkhorn distance is the Frobenius inner product of PI and C
        emd_dist = torch.sum(PI * C, dim=(-2, -1))

        if self.reduction == "mean":
            return emd_dist.mean()
        elif self.reduction == "sum":
            return emd_dist.sum()
        return emd_dist

    def _M(self, C, u, v):
        """Modified cost matrix for stability"""
        return u.unsqueeze(-1) + v.unsqueeze(-2) - C
