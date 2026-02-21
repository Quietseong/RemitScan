import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc21 = nn.Linear(16, 8)   # mu
        self.fc22 = nn.Linear(16, 8)   # logvar
        self.fc3 = nn.Linear(8, 16)
        self.fc4 = nn.Linear(16, input_dim)

    def encode(self, x: torch.Tensor):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld
