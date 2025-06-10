from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class ApiSettings(BaseSettings):
    """
    Defines the API server settings.
    """
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["*"]

class DatabaseSettings(BaseSettings):
    """
    Defines database connection settings.
    """
    USER: str
    PASSWORD: str
    HOST: str
    PORT: int
    NAME: str

    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        return f"postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"

    # Prefix all environment variables with `DB_`
    model_config = SettingsConfigDict(env_prefix='DB_')


class AuthSettings(BaseSettings):
    """
    Defines authentication settings, particularly for JWT.
    """
    SECRET_KEY: str = "a_very_secret_key_that_should_be_in_an_env_file"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Prefix all environment variables with `AUTH_`
    model_config = SettingsConfigDict(env_prefix='AUTH_')


class CelerySettings(BaseSettings):
    """
    Defines Celery worker settings.
    """
    BROKER_URL: str = "redis://localhost:6379/0"
    RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Prefix all environment variables with `CELERY_`
    model_config = SettingsConfigDict(env_prefix='CELERY_')


class Settings(BaseSettings):
    """
    Main settings class that aggregates all other settings.
    Loads variables from a .env file.
    """
    api: ApiSettings = ApiSettings()
    db: DatabaseSettings = DatabaseSettings()
    auth: AuthSettings = AuthSettings()
    celery: CelerySettings = CelerySettings()
    
    # This tells pydantic-settings to load variables from a .env file
    # and allows nested models (like ApiSettings and DatabaseSettings)
    model_config = SettingsConfigDict(env_file='.env', env_nested_delimiter='__')

# Create a single, project-wide instance of the settings
settings = Settings() 