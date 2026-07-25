from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    algorithm: str = Field(..., alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(..., alias="JWT_EXPIRE_MINUTES")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()