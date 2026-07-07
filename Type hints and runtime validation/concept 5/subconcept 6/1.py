from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Literal

class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd", "adamw"]
    lr: float = Field(gt = 1e-6, lt = 1.0 )
    weight_decay: float = Field(default = 0.0, ge = 0)
    
class SchedulerConfig(BaseModel):
    name: Literal["cosine", "linear", "constant"]
    max_steps: int = Field(gt = 0)
    warmup_steps: int = Field(default = 0, ge = 0)

    
class TrainingConfig(BaseModel):
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    epochs: int = Field(gt = 0)
    batch_size: int = Field(gt = 0)
    
    @model_validator(mode="after")
    def step_validator(self)->"TrainingConfig":
        if self.scheduler.warmup_steps >= self.scheduler.max_steps:
            raise ValueError(
            f"warmup_steps ({self.scheduler.warmup_steps}) must be "
            f"less than max_steps ({self.scheduler.max_steps})"
            )
        
        return self
    
def demonstrate(data: dict)->TrainingConfig:
    config = TrainingConfig.model_validate(data) 
    print(config)
    return config
    
valid_data = {
    "optimizer": {
        "name": "adam",
        "lr": 0.001,
        "weight_decay": 0.01
    },
    "scheduler": {
        "name": "cosine",
        "max_steps": 1000,
        "warmup_steps": 100
    },
    "epochs": 10,
    "batch_size": 32
}

invalid_data = {
    "optimizer": {
        "name": "adam",
        "lr": 0.001,
        "weight_decay": 0.01
    },
    "scheduler": {
        "name": "cosine",
        "max_steps": 100,
        "warmup_steps": 200
    },
    "epochs": 10,
    "batch_size": 32
}

demonstrate(valid_data)

try:
    demonstrate(invalid_data)
except ValidationError as e:
    print(f"Validation error: {e}")