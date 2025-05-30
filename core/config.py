"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv
from .exceptions import ConfigurationError

class ConfigManager:
    """Manages application configuration from multiple sources."""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv('APP_ENV', 'development')
        self.config_dir = Path(__file__).parent.parent / 'config'
        self.config: Dict[str, Any] = {}
        
        # Load configurations in order
        self._load_env_vars()
        self._load_base_config()
        self._load_env_config()
        self._validate_config()
    
    def _load_env_vars(self) -> None:
        """Load environment variables from .env file."""
        env_file = self.config_dir / '.env'
        if env_file.exists():
            load_dotenv(env_file)
        
    def _load_base_config(self) -> None:
        """Load base configuration from yaml file."""
        try:
            with open(self.config_dir / 'base.yaml', 'r') as f:
                self.config.update(yaml.safe_load(f))
        except Exception as e:
            raise ConfigurationError(f"Failed to load base config: {str(e)}")
    
    def _load_env_config(self) -> None:
        """Load environment-specific configuration."""
        env_config_file = self.config_dir / f'{self.env}.yaml'
        if env_config_file.exists():
            try:
                with open(env_config_file, 'r') as f:
                    env_config = yaml.safe_load(f)
                    self._deep_update(self.config, env_config)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load {self.env} config: {str(e)}"
                )
    
    def _deep_update(self, d: dict, u: dict) -> None:
        """Recursively update nested dictionaries."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
    
    def _validate_config(self) -> None:
        """Validate required configuration values."""
        required_keys = [
            'api.alpaca.key',
            'api.alpaca.secret',
            'api.alpha_vantage.key',
            'database.url',
            'logging.level'
        ]
        
        for key in required_keys:
            if not self.get(key):
                raise ConfigurationError(f"Missing required config: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('app.debug', False)

# Global configuration instance
config = ConfigManager() 