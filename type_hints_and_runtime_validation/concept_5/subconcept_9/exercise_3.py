from dataclasses import dataclass, field
from typing import TypedDict

from pydantic import BaseModel, Field, ValidationError, model_validator


class ModelRegistryPayload(TypedDict):
    model_name: str
    model_version: str
    hyperparameters: dict[str, float]
    metrics: dict[str, float]
    tags: list[str]


class ModelRegistryConfig(BaseModel):
    model_name: str
    model_version: str
    hyperparameters: dict[str, float]
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_metrics_and_version(self):
        if not self.model_version.startswith("v"):
            raise ValueError("model_version must start with 'v'")

        if "accuracy" in self.metrics and not (0.0 <= self.metrics["accuracy"] <= 1.0):
            raise ValueError("accuracy must be between 0.0 and 1.0")

        return self

    @classmethod
    def from_payload(cls, payload: ModelRegistryPayload) -> "ModelRegistryConfig":
        return cls.model_validate(payload)

    def primary_metrics(self) -> dict[str, float]:
        primary_keys = {"accuracy", "loss", "f1"}
        return {k: v for k, v in self.metrics.items() if k in primary_keys}

    def describe(self) -> str:
        metrics_str = ", ".join(
            f"{k}={v:.3f}" for k, v in self.primary_metrics().items()
        )
        return f"Model: {self.model_name} {self.model_version} | Metrics: {metrics_str}"


@dataclass
class RegisteredModel:
    config: ModelRegistryConfig
    registry_id: str
    size_mb: float
    warnings: list[str] = field(default_factory=list)


def register_model(
    payload: ModelRegistryPayload,
    registry_id: str,
    size_mb: float,
) -> RegisteredModel:
    config = ModelRegistryConfig.from_payload(payload)
    warnings = []

    if size_mb > 1000:
        warnings.append(f"Warning: model size {size_mb}MB exceeds 1GB")

    if not config.metrics:
        warnings.append("Warning: no metrics provided")

    return RegisteredModel(
        config=config,
        registry_id=registry_id,
        size_mb=size_mb,
        warnings=warnings,
    )


valid_payload: ModelRegistryPayload = {
    "model_name": "fraud-detector",
    "model_version": "v1.2.0",
    "hyperparameters": {"lr": 0.001, "dropout": 0.3},
    "metrics": {"accuracy": 0.94, "loss": 0.12, "f1": 0.91},
    "tags": ["production", "fraud"],
}

result = register_model(valid_payload, registry_id="reg-001", size_mb=450.0)
print(result.config.describe())

oversized_payload: ModelRegistryPayload = {
    "model_name": "large-embedder",
    "model_version": "v0.1.0",
    "hyperparameters": {"layers": 48.0, "hidden_dim": 1024.0},
    "metrics": {},
    "tags": [],
}

result_2 = register_model(oversized_payload, registry_id="reg-002", size_mb=1500.0)
print(result_2.warnings)