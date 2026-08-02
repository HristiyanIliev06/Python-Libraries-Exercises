import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseModel):
    bucket_name: str = Field(pattern = r"^[a-z0-9\-]+$") 
    credentials_path: str  = Field(pattern=r'^/.+')
    max_upload_size_mb: int = Field(default = 500, ge = 1, le = 10000)

class MLPipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)
    environment: Literal["local", "staging", "production"] 
    database_url: str = Field(pattern=r'^postgresql://.+')
    storage: StorageSettings
    workers: int = 1
    enable_profiling: bool = False
    alert_webhook_url: str | None = None

    @model_validator(mode="after")
    def validate_workers_enable_profiling_alert_webhook_url(self):
        if self.environment == "production":
            if self.workers < 4:
                raise ValueError("Workers' number must be at least 4 in production!")

            if self.enable_profiling == True:
                raise ValueError("Profiling overhead forbidden in production!")

        if (self.environment == "production" or self.environment == "staging") and self.alert_webhook_url in (None, ""):
            raise ValueError("Alert webhook url required in production or staging!")

        return self

def get_pipeline_settings(env_overrides: dict | None = None) -> MLPipelineSettings:
    injected = []
    for key, value in env_overrides.items():
        if value is not None:
            key: str
            os.environ[key.upper()] = str(value)
            injected.append(key.upper())
    try:
        return MLPipelineSettings()
    finally:
        for key in injected:
            os.environ.pop(key, None)

"""Write four test calls via get_pipeline_settings:

Valid local config — minimal fields, no webhook, 1 worker
Valid production config — workers=4, enable_profiling=False, webhook set, storage with valid bucket
Production config that fails — enable_profiling=True
Staging config that fails — alert_webhook_url is None

Each failure case should be wrapped in try/except and print the validation error.

Constraints:

storage field receives a JSON string from env (STORAGE='{"bucket_name": "ml-data", ...}') — Pydantic handles JSON coercion for nested models in BaseSettings automatically, but you need to understand why
get_pipeline_settings must clean up env vars even if instantiation raises — think about the right construct for that
No model_validate — all instantiation goes through get_pipeline_settings"""

valid_local_config = {
    "environment": "local",
    "database_url": "postgresql://localhost:5432/mydb",
    "storage": '{"bucket_name": "ml-data", "credentials_path": "/path/to/creds.json", "max_upload_size_mb": 500}',
    "workers": 1,
    "enable_profiling": False,
    "alert_webhook_url": None
}

print(get_pipeline_settings(valid_local_config))

valid_production_config = {
    "environment": "production",
    "database_url": "postgresql://prod-db:5432/mydb",
    "storage": '{"bucket_name": "ml-data", "credentials_path": "/path/to/creds.json", "max_upload_size_mb": 500}',
    "workers": 4,
    "enable_profiling": False,
    "alert_webhook_url": "https://hooks.example.com/alert"
}

print(get_pipeline_settings(valid_production_config))

invalid_production_config = {
    "environment": "production",
    "database_url": "postgresql://prod-db:5432/mydb",
    "storage": '{"bucket_name": "ml-data", "credentials_path": "/path/to/creds.json", "max_upload_size_mb": 500}',
    "workers": 4,
    "enable_profiling": True,
    "alert_webhook_url": "https://hooks.example.com/alert"
}

try:
    print(get_pipeline_settings(invalid_production_config))
except ValidationError as e:
    print(f"Error occurred: {e}")

invalid_staging_config = {
    "environment": "staging",
    "database_url": "postgresql://staging-db:5432/mydb",
    "storage": '{"bucket_name": "ml-data", "credentials_path": "/path/to/creds.json", "max_upload_size_mb": 500}',
    "workers": 2,
    "enable_profiling": False,
    "alert_webhook_url": None
}

try:
    print(get_pipeline_settings(invalid_staging_config))
except ValidationError as e:
    print(f"Error occurred: {e}")
    

    