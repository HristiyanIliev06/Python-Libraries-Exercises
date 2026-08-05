from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field, ValidationError


class RawExperimentPayload(TypedDict):
    experiment_name: str
    learning_rate: float
    epochs: int
    tags: list[str]

class ValidatedExperimentConfig(BaseModel):
    experiment_name: str
    learning_rate: float = Field(ge = 1e-6, le = 1.0)
    epochs: int = Field(ge = 1, le = 1000)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: RawExperimentPayload)->"ValidatedExperimentConfig":
        return cls.model_validate(payload)

@dataclass
class PipelineResult:
    config: ValidatedExperimentConfig
    run_id: str
    elapsed_seconds: float
    artifact_paths: list[str]

""""Write two test calls:

Build a ValidatedExperimentConfig from a valid RawExperimentPayload via from_payload
Show what happens when learning_rate is out of range — catch and print the error
"""

# Test call 1: Valid RawExperimentPayload
valid_payload: RawExperimentPayload = {
    "experiment_name": "test_experiment",
    "learning_rate": 0.01,
    "epochs": 10,
    "tags": ["test", "experiment"]
}

print("Test Call 1: Valid Payload")
config_1 = ValidatedExperimentConfig.from_payload(valid_payload)
print(f"Validated Config 1: {config_1}")

# Test call 2: Invalid RawExperimentPayload (learning_rate out of range)
invalid_payload: RawExperimentPayload = {
    "experiment_name": "test_experiment_invalid",
    "learning_rate": 1.5,  # Out of range
    "epochs": 10,
    "tags": ["test", "experiment"]
}

print("\nTest Call 2: Invalid Payload (learning_rate out of range)")
try:
    config_2 = ValidatedExperimentConfig.from_payload(invalid_payload)
    print(f"Validated Config 2: {config_2}")
except ValidationError as e:
    print(f"Error: {e}")