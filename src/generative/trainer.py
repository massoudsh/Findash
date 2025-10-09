import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from typing import Dict

from .models import VAE, Generator, Discriminator

logger = logging.getLogger(__name__)

class VAEGANTrainer:
    """
    A trainer class to handle the training loop of the VAE-GAN model.
    """
    def __init__(self, config: Dict):
        """
        Initializes the trainer with models, optimizers, and a configuration.
        Args:
            config (Dict): A dictionary containing model dimensions, hyperparameters,
                           and device information.
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.vae = VAE(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim']
        ).to(self.device)
        self.generator = Generator(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['input_dim']
        ).to(self.device)
        self.discriminator = Discriminator(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)

        # Initialize optimizers
        lr = config['learning_rate']
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def train(self, dataloader: DataLoader, num_epochs: int):
        """
        Runs the main training loop for the specified number of epochs.
        Args:
            dataloader (DataLoader): DataLoader providing the training data.
            num_epochs (int): The number of epochs to train for.
        """
        logger.info(f"Starting VAE-GAN training for {num_epochs} epochs on {self.device}...")
        for epoch in range(num_epochs):
            for i, (data,) in enumerate(dataloader):
                real_data = data.to(self.device)
                batch_size = real_data.size(0)

                # Train VAE
                reconstructed, mean, logvar = self.vae(real_data)
                vae_loss = self._calculate_vae_loss(reconstructed, real_data, mean, logvar)
                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                self.vae_optimizer.step()

                # Train Discriminator
                with torch.no_grad():
                    z = self.vae.reparameterize(mean, logvar)
                    fake_data = self.generator(z)
                disc_loss = self._train_discriminator_step(real_data, fake_data, batch_size)

                # Train Generator
                gen_loss = self._train_generator_step(fake_data, batch_size)
            
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"VAE Loss: {vae_loss.item():.4f}, "
                f"Disc Loss: {disc_loss.item():.4f}, "
                f"Gen Loss: {gen_loss.item():.4f}"
            )
        logger.info("Training finished.")

    def _calculate_vae_loss(self, recon, real, mean, logvar):
        recon_loss = self.mse_loss(recon, real)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def _train_discriminator_step(self, real_data, fake_data, batch_size):
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        real_preds = self.discriminator(real_data)
        fake_preds = self.discriminator(fake_data)
        
        disc_loss = (self.bce_loss(real_preds, real_labels) + self.bce_loss(fake_preds, fake_labels)) / 2
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        return disc_loss

    def _train_generator_step(self, fake_data, batch_size):
        real_labels = torch.ones(batch_size, 1).to(self.device)
        
        # We need to re-evaluate the discriminator on the fake data after its update
        fake_preds = self.discriminator(fake_data)
        gen_loss = self.bce_loss(fake_preds, real_labels)
        
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()
        return gen_loss

    def save_models(self, path: str = 'models/generative'):
        """Saves the trained models to the specified directory."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.vae.state_dict(), f"{path}/vae.pth")
        torch.save(self.generator.state_dict(), f"{path}/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{path}/discriminator.pth")
        logger.info(f"Models saved to {path}") 