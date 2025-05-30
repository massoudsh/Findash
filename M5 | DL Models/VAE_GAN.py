# Library imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    """Variational Autoencoder (VAE) implementation.
    
    This neural network creates a lower-dimensional latent representation of the input data
    while maintaining its key features and distribution properties.
    
    Args:
        input_dim (int): Dimension of input data
        hidden_dim (int): Dimension of hidden layers
        latent_dim (int): Dimension of latent space
    """
    # ... existing __init__ ...

    def encode(self, x):
        """Encode input data into latent space parameters.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            tuple: (mean, logvar) of the latent space distribution
        """
        h = F.relu(self.fc1(x))
        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mean, logvar):
        """Perform the reparameterization trick for enabling backpropagation.
        
        Args:
            mean (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            
        Returns:
            torch.Tensor: Sampled point from the latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

# ... existing decode and forward methods ...

class Generator(nn.Module):
    """Generator network for the GAN component.
    
    Transforms random noise from the latent space into synthetic data samples.
    
    Args:
        latent_dim (int): Dimension of input noise vector
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of generated output
    """
    # ... existing implementation ...

class Discriminator(nn.Module):
    """Discriminator network for the GAN component.
    
    Attempts to distinguish between real and generated (fake) samples.
    
    Args:
        input_dim (int): Dimension of input data
        hidden_dim (int): Dimension of hidden layers
    """
    # ... existing implementation ...

# Training Configuration
# ------------------------
# Model Architecture Parameters
input_dim = 28*28  # Flattened input dimension
hidden_dim = 256   # Hidden layer dimension
latent_dim = 64    # Latent space dimension

# Training Hyperparameters
batch_size = 64    # Number of samples per batch
lr = 0.001        # Learning rate for all optimizers
num_epochs = 50    # Total training epochs

# Initialize Models
vae = VAE(input_dim, hidden_dim, latent_dim)
generator = Generator(latent_dim, hidden_dim, input_dim)
discriminator = Discriminator(input_dim, hidden_dim)

# Initialize Optimizers
vae_optimizer = optim.Adam(vae.parameters(), lr=lr)
gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Loss Functions
bce_loss = nn.BCELoss()    # Binary Cross Entropy for GAN
mse_loss = nn.MSELoss()    # Mean Squared Error for VAE reconstruction

# Training Loop
# ------------------------
for epoch in range(num_epochs):
    for batch in dataloader:
        real_data = batch[0]
        
        # Phase 1: Train VAE
        # -----------------
        reconstructed, mean, logvar = vae(real_data)
        recon_loss = mse_loss(reconstructed, real_data)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_divergence
        
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

        # Phase 2: Train Discriminator
        # --------------------------
        z = vae.reparameterize(mean, logvar)
        fake_data = generator(z)
        
        real_labels = torch.ones(real_data.size(0), 1)
        fake_labels = torch.zeros(fake_data.size(0), 1)

        # Calculate discriminator loss on real and fake data
        real_preds = discriminator(real_data)
        fake_preds = discriminator(fake_data.detach())
        disc_loss = (bce_loss(real_preds, real_labels) + 
                    bce_loss(fake_preds, fake_labels)) / 2

        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # Phase 3: Train Generator
        # ----------------------
        fake_preds = discriminator(fake_data)
        gen_loss = bce_loss(fake_preds, real_labels)

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

    # Print training progress
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"VAE Loss: {vae_loss.item():.4f} | "
          f"Disc Loss: {disc_loss.item():.4f} | "
          f"Gen Loss: {gen_loss.item():.4f}") 