from src.core.celery_app import celery_app
from .service import train_generative_model, generate_synthetic_data
import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Default configuration for the VAE-GAN model
# This should be moved to a proper configuration management system
DEFAULT_CONFIG = {
    'input_dim': 60,       # e.g., 60 days of price data
    'hidden_dim': 128,
    'latent_dim': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'model_save_path': 'models/generative'
}

@celery_app.task(name='generative.train_model')
def train_model_task(data_path: str, config: Dict = None):
    """
    Celery task to train the generative VAE-GAN model.
    Args:
        data_path (str): Path to the CSV file containing historical financial data.
        config (Dict, optional): Configuration overrides for the model and training.
    """
    logger.info(f"Received task to train generative model with data from {data_path}")
    try:
        # Load data
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)[['Close']]
        
        # Train model
        train_config = {**DEFAULT_CONFIG, **(config or {})}
        result = train_generative_model(data, train_config)
        
        return result
    except Exception as e:
        logger.error(f"Generative model training task failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@celery_app.task(name='generative.generate_data')
def generate_data_task(num_samples: int, config: Dict = None, model_path: str = None) -> Dict:
    """
    Celery task to generate synthetic data samples.
    Args:
        num_samples (int): The number of synthetic data samples to generate.
        config (Dict, optional): Configuration overrides.
        model_path (str, optional): Path to the trained models.
    """
    logger.info(f"Received task to generate {num_samples} synthetic samples.")
    try:
        gen_config = {**DEFAULT_CONFIG, **(config or {})}
        model_path = model_path or gen_config['model_save_path']

        synthetic_data = generate_synthetic_data(num_samples, gen_config, model_path)
        
        # Here you might save the data to a file or database
        output_path = f"data/synthetic_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(output_path, synthetic_data)

        return {"status": "success", "samples_generated": num_samples, "output_path": output_path}
    except Exception as e:
        logger.error(f"Synthetic data generation task failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)} 