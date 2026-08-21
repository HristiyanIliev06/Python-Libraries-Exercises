from typing import Literal
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class TrainingServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file="training.env", env_file_encoding="utf-8", case_sensitive=False)
    experiment_tracker_url: str 
    model_registry_path: str 
    max_parallel_jobs: int = 2
    enable_gpu: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("experiment_tracker_url")
    @classmethod
    def experiment_tracker_url_format_validator(cls, v: str):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Experiment tracker url must begin with 'http://' or 'https://'!")

        return v


"""Write three instantiation calls:
Pass all required fields as kwargs (no .env file needed)
Simulate env var injection using os.environ patching before instantiation
Show what happens when experiment_tracker_url fails validation — catch and print the error"""

training_service_settings_1 = TrainingServiceSettings(
    experiment_tracker_url="http://my-tracker.com",
    model_registry_path="/path/to/registry",
    max_parallel_jobs=2,
    enable_gpu=False,
    log_level="INFO"
)


os.environ["EXPERIMENT_TRACKER_URL"] = "http://my-tracker.com"
os.environ["MODEL_REGISTRY_PATH"] = "/path/to/registry"
training_service_settings_2 = TrainingServiceSettings()

try:
    training_service_settings_3 = TrainingServiceSettings(
        experiment_tracker_url="http:/my-tracker.com",
        model_registry_path="/path/to/registry",
        max_parallel_jobs=2,
        enable_gpu=False,
        log_level="INFO"
)
except ValidationError as e:
    print(f"Error occurred: {e}")