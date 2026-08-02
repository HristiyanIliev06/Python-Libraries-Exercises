from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Literal, Optional
#import math

#<From 1.py> =========================================================
class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd", "adamw"]
    lr: float = Field(gt = 1e-6, lt = 1.0 )
    weight_decay: float = Field(default = 0.0, ge = 0)
    
class SchedulerConfig(BaseModel):
    name: Literal["cosine", "linear", "constant"]
    max_steps: int = Field(gt = 0)
    warmup_steps: int = Field(default = 0, ge = 0)

#=====================================================================


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
        
# Sb 6 ^====================================================================================^    

def serialize_config(config: ExperimentConfig) -> str:
    return config.model_dump_json()

def deserialize_config(raw: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(raw)

config = {
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


config1 = ExperimentConfig.model_validate(config)
serialized = serialize_config(config1)
config2 = deserialize_config(serialized)
assert config1 == config2  # model instance vs model instance

print(ExperimentConfig.model_validate(config).model_dump(exclude_none=True, exclude_defaults=True))