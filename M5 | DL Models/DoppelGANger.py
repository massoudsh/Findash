import pandas as pd
import torch
import tensorflow as tf
from gretel_synthetics import TimeSeries
from fingan import FinGAN
from timegan import TimeGAN
from diffusion_model import FinancialDiffusionModel

class FinancialSynthesizer:
    def __init__(self, model_type='timegan'):
        """
        Initialize the synthesizer with specified model type
        Available models: 'gretel', 'fingan', 'timegan', 'diffusion'
        """
        self.model_type = model_type
        self.model = None
    
    def load_data(self, data_source):
        """Unified data loading method"""
        if isinstance(data_source, str):
            return pd.read_csv(data_source)
        return data_source

    def train(self, data, **kwargs):
        """Train the selected model"""
        data = self.load_data(data)
        
        if self.model_type == 'gretel':
            config = TimeSeries.ModelConfig(
                max_sequence_len=kwargs.get('max_seq_len', 550),
                sample_len=kwargs.get('sample_len', 550),
                batch_size=kwargs.get('batch_size', 100),
                epochs=kwargs.get('epochs', 1000)
            )
            self.model = TimeSeries.train(data, config)

        elif self.model_type == 'fingan':
            self.model = FinGAN(input_dim=data.shape[1])
            self.model.train(data, 
                           epochs=kwargs.get('epochs', 1000),
                           batch_size=kwargs.get('batch_size', 64))

        elif self.model_type == 'timegan':
            parameters = {
                "module": kwargs.get('module', 'gru'),
                "hidden_dim": kwargs.get('hidden_dim', 24),
                "num_layer": kwargs.get('num_layer', 3),
                "iterations": kwargs.get('iterations', 10000),
                "batch_size": kwargs.get('batch_size', 128),
            }
            self.model = TimeGAN(parameters)
            self.model.train(data)

        elif self.model_type == 'diffusion':
            preprocessed_data = preprocess_financial_data(data)
            self.model = FinancialDiffusionModel()
            self.model.train(preprocessed_data, epochs=kwargs.get('epochs', 1000))

    def generate(self, num_samples):
        """Generate synthetic data using the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before generating data")

        if self.model_type == 'gretel':
            return self.model.generate(num_records=num_samples)
        elif self.model_type == 'fingan':
            return self.model.generate(num_samples=num_samples)
        elif self.model_type == 'timegan':
            return self.model.generate(num_samples)
        elif self.model_type == 'diffusion':
            synthetic_images = self.model.generate(num_samples=num_samples)
            return inverse_transform(synthetic_images)

# Example usage:
# synthesizer = FinancialSynthesizer(model_type='timegan')
# synthesizer.train(data='financial_data.csv', epochs=1000)
# synthetic_data = synthesizer.generate(num_samples=1000)