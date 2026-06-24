from pydantic import BaseModel
from typing import Literal

class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd", "adamw"]
    weight_decay: float = 0.0

class ExperimentConfig(BaseModel):
    experiment_name: str
    optimizer: OptimizerConfig
    learning_rate: float
    batch_size: int
    epochs: int = 10    #There is no point in setting a default value to string of a number, if a variable will accept integers anyway.
    device: Literal["cpu", "cuda"] = "cpu"
    tags: list[str] = list()

config = ExperimentConfig(
    experiment_name="run_01",
    optimizer= {"name": "adam", "weight_decay": 0.01},
    learning_rate=3e-4,
    batch_size=32,
    epochs=5,
    device="cuda",
    tags=["v1", "baseline"]
)

print(config.optimizer.name)
