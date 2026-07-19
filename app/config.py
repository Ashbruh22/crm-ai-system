from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SECRET_KEY: str
    DATABASE_URL: str
    REDIS_URL: str
    WEB_CONCURRENCY: int = 4
    DOCS_ENABLED: bool = True
    BATCH_CONCURRENCY: int = 10
    
    # CRM Integration Settings
    CRM_WEBHOOK_SECRET: str = "your-hmac-secret-here"
    CRM_SIMULATE_ENABLED: bool = True
    
    # Allows reading from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
