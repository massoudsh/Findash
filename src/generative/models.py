import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for learning a latent representation of sequential data.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # Use sigmoid for normalized data

    def forward(self, x):
        mean, logvar = self.encode(x.view(-1, self.fc1.in_features))
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

class Generator(nn.Module):
    """
    Generator network for the GAN component.
    Transforms a latent vector into a synthetic data sequence.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

class Discriminator(nn.Module):
    """
    Discriminator network for the GAN component.
    Distinguishes between real and generated data sequences.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        return torch.sigmoid(self.fc3(h)) 