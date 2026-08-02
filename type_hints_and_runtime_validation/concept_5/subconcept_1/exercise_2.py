from typing import Literal
from pydantic import BaseModel, ValidationError
import json
    
class ExperimentConfig(BaseModel):
    experiment_name: str
    model_type: Literal["transformer", "lstm", "cnn"]
    learning_rate: float
    batch_size: int
    epochs: int
    device: Literal["cpu", "cuda"]
    tags: list[str] = []
    
def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = json.load(f)
    return ExperimentConfig(**raw)

def summarize_config(config: ExperimentConfig) -> str:
    return f"""Experiment '{config.experiment_name}' using 
    {config.model_type} on {config.device} for {config.epochs} epochs
    with learning rate {config.learning_rate} and {len(config.tags)} tags."""

path_to_valid = "Type hints and runtime validation/concept 5/subconcept 1/files/valid.json"
path_to_invalid = "Type hints and runtime validation/concept 5/subconcept 1/files/invalid.json"

print(summarize_config(load_experiment_config(path_to_valid)))

try:
    print(summarize_config(load_experiment_config(path_to_invalid)))
except ValidationError as e:
    print(f"Error loading experiment config: {e}")


