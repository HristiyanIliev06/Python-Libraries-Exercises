import os
from typing import Literal
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResourceLimits(BaseModel):
    max_memory_gb: int = Field(default=8, ge=1, le=256)
    max_cpu_cores: int = Field(default=4, ge=1, le=64)
    gpu_enabled: bool = False


class InferenceServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="inference.env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    service_name: str
    model_endpoint: str
    deployment_env: Literal["dev", "staging", "production"]
    resources: ResourceLimits
    request_timeout_ms: int = Field(default=5000, ge=100)
    enable_caching: bool = True
    fallback_endpoint: str | None = None

    @model_validator(mode="after")
    def validate_production_constraints(self):
        if self.deployment_env == "production":
            if self.resources.max_memory_gb < 16:
                raise ValueError("Production requires at least 16GB memory")
            if self.fallback_endpoint is None:
                raise ValueError("Production requires a fallback endpoint")

        if self.deployment_env in ("staging", "production"):
            if not self.model_endpoint.startswith("https://"):
                raise ValueError("Staging and production require HTTPS endpoints")

        return self # Must return self to ensure the model_validator works correctly


def load_inference_settings(overrides: dict | None = None) -> InferenceServiceSettings:
    injected = []
    if overrides is not None:
        for key, value in overrides.items():
            if value is not None:
                os.environ[key.upper()] = str(value)
                injected.append(key.upper())

    try:                        # Ensures that environment variables are cleaned up after instantiation no matter what                
        settings = InferenceServiceSettings()
        return settings
    finally:
        for key in injected:
            os.environ.pop(key, None)


valid_dev = load_inference_settings({
    "service_name": "inference-dev",
    "model_endpoint": "http://dev-model:8080/predict",
    "deployment_env": "dev",
    "resources": '{"max_memory_gb": 4, "max_cpu_cores": 2, "gpu_enabled": false}',
    "request_timeout_ms": 3000,
})

print(valid_dev)

try:
    invalid_staging = load_inference_settings({
        "service_name": "inference-staging",
        "model_endpoint": "http://staging-model:8080/predict",
        "deployment_env": "staging",
        "resources": '{"max_memory_gb": 8, "max_cpu_cores": 4, "gpu_enabled": false}',
    })
except ValidationError as e:
    print(f"Validation error: {e}")