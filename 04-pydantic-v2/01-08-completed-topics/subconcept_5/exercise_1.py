from pydantic import BaseModel, Field
from typing import Literal, Union, Annotated

class AdamConfig(BaseModel):
    optimizer: Literal["adam"]
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999

class SGDConfig(BaseModel):
    optimizer: Literal["sgd"]
    lr: float
    momentum: float = 0.0

class AdamWConfig(BaseModel):
    optimizer: Literal["adamw"]
    lr: float
    weight_decay: float = 0.01

OptimizerConfig = Union[AdamConfig, SGDConfig, AdamWConfig]

class TrainingConfig(BaseModel):
    optimizer_config: Annotated[OptimizerConfig, Field(discriminator = "optimizer")]
    
tc = TrainingConfig(optimizer_config={"optimizer": "adam", "lr": 0.001})
tc2 = TrainingConfig(optimizer_config={"optimizer": "sgd", "lr": 0.01, "momentum": 0.9})
tc3 = TrainingConfig(optimizer_config={"optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01})

print(type(tc.optimizer_config))
print(type(tc2.optimizer_config))
print(type(tc3.optimizer_config))

try:
    tc4 = TrainingConfig(optimizer_config={"optimizer": "rmsprop", "lr": 0.001})
except Exception as e:
    print(e)
