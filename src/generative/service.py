import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
import logging
import os

from .models import Generator
from .trainer import VAEGANTrainer

logger = logging.getLogger(__name__)

# --- Data Preparation ---

def prepare_financial_data(data: pd.DataFrame, sequence_length: int) -> DataLoader:
    """
    Prepares financial time-series data for the VAE-GAN.
    - Normalizes the data.
    - Creates sequences of a specified length.
    - Returns a DataLoader.
    """
    # Simple min-max scaling
    data_normalized = (data - data.min()) / (data.max() - data.min())
    sequences = []
    for i in range(len(data_normalized) - sequence_length + 1):
        sequences.append(data_normalized.iloc[i:i+sequence_length].values)
    
    sequences_tensor = torch.FloatTensor(np.array(sequences))
    dataset = TensorDataset(sequences_tensor)
    return DataLoader(dataset, batch_size=64, shuffle=True)

# --- Training Service ---

def train_generative_model(data: pd.DataFrame, config: Dict) -> Dict:
    """
    Orchestrates the training of the VAE-GAN model.
    """
    logger.info("Starting generative model training service...")
    sequence_length = config['input_dim']
    dataloader = prepare_financial_data(data, sequence_length)
    
    trainer = VAEGANTrainer(config)
    trainer.train(dataloader, num_epochs=config['num_epochs'])
    
    model_path = config.get('model_save_path', 'models/generative')
    trainer.save_models(model_path)
    
    logger.info("Generative model training complete.")
    return {"status": "success", "model_path": model_path}

# --- Generation Service ---

def generate_synthetic_data(num_samples: int, config: Dict, model_path: str) -> np.ndarray:
    """
    Generates synthetic data samples using a pre-trained Generator model.
    """
    logger.info(f"Generating {num_samples} synthetic data samples...")
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['input_dim']
    ).to(device)
    
    generator_path = os.path.join(model_path, 'generator.pth')
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator model not found at {generator_path}")
        
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, config['latent_dim']).to(device)
        synthetic_data = generator(noise)
    
    logger.info("Synthetic data generation complete.")
    return synthetic_data.cpu().numpy() 