from pydantic_settings import BaseSettings
from typing import List
import os

from pydantic.type_adapter import P

class Settings(BaseSettings):
    PROJECT_NAME: str = "Meeting Intelligence"
    SQLALCHEMY_DATABASE_URI: str 
    REDIS_URL: str = "redis://redis:6379/0"
    SECRET_KEY: str
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./data/uploads")
    ENV : str = "development"  
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

