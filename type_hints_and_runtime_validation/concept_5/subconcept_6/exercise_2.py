from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Literal, Optional
#import math
import json
from pathlib import Path
from subconcept_6.exercise_1 import OptimizerConfig, SchedulerConfig


class DataConfig(BaseModel):
    train_path: str = Field(pattern = r".+\.(csv|jsonl)$")
    val_path: str = Field(pattern = r".+\.(csv|jsonl)$")
    max_samples: Optional[int] = Field(gt = 0, default = None)
    num_workers: int = Field(ge = 0, le = 16, default = 4)
    
class ModelConfig(BaseModel):
    architecture: Literal["transformer", "lstm", "cnn"]
    hidden_dim: int  = Field(gt = 0)
    num_layers: int = Field(ge = 1, default = 2)
    dropout: float = Field(ge = 0.0, lt = 1.0, default = 0.1)
    
    @field_validator("hidden_dim")
    @classmethod
    def hidden_dim_validator(cls, v: int) -> int:
        #if not math.log2(v) % 2 in (0.0, 1.0):
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError(f"hidden_dim must be a positive power of 2, got {v}")
        return v
    
class ExperimentConfig(BaseModel):
    experiment_name: str = Field(min_length = 3, max_length = 64) 
    device: Literal["cpu", "cuda", "mps"]
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    
    @model_validator(mode = "after")
    def hardware_support_validator(self)->"ExperimentConfig":
        if self.device == "mps" and self.model.architecture == "lstm":
            raise ValueError("MPS doesn't support LSTM in this system")
        
        return self
    
def load_experiment_config(path: str) -> ExperimentConfig:
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    return ExperimentConfig.model_validate(raw)

valid_experiment = {
    "experiment_name": "my_experiment",
    "device": "cuda",
    "seed": 42,
    "data": {
        "train_path": "train.csv",
        "val_path": "val.csv",
        "max_samples": 1000,
        "num_workers": 4
    },
    "model": {
        "architecture": "transformer",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1
    },
    "optimizer": {
        "name": "adam",
        "lr": 0.001,
        "weight_decay": 0.01
    },
    "scheduler": {
        "name": "cosine",
        "max_steps": 1000,
        "warmup_steps": 100
    }
}

invalid_experiment_mps_lstm = {
    "experiment_name": "my_experiment",
    "device": "mps",
    "seed": 42,
    "data": {
        "train_path": "train.csv",
        "val_path": "val.csv",
        "max_samples": 1000,
        "num_workers": 4
    },
    "model": {
        "architecture": "lstm",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1
    },
    "optimizer": {
        "name": "adam",
        "lr": 0.001,
        "weight_decay": 0.01
    },
    "scheduler": {
        "name": "cosine",
        "max_steps": 1000,
        "warmup_steps": 100
    }
}

invalid_experiment_nested_field = {
    "experiment_name": "my_experiment",
    "device": "cuda",
    "seed": 42,
    "data": {
        "train_path": "train.csv",
        "val_path": "val.csv",
        "max_samples": -1000,  # Invalid: max_samples must be > 0
        "num_workers": 4
    },
    "model": {
        "architecture": "transformer",
        "hidden_dim": 127, # Invalid: hidden_dim must be a power of 2
        "num_layers": 2,
        "dropout": 0.1
    },
    "optimizer": {
        "name": "adam",
        "lr": 0.001,
        "weight_decay": 0.01
    },
    "scheduler": {
        "name": "cosine",
        "max_steps": 1000,
        "warmup_steps": 100
    }
}

print("Valid experiment config:")
print(ExperimentConfig.model_validate(valid_experiment))

try:
    print(ExperimentConfig.model_validate(invalid_experiment_mps_lstm))
except ValidationError as e:
    print("MPS/LSTM guard triggered:", e)
    
try:
    print(ExperimentConfig.model_validate(invalid_experiment_nested_field))
except ValidationError as e:
    print("Nested field errors triggered:", e)