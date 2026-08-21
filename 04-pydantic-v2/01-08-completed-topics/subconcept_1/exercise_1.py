from pydantic import BaseModel
from typing import Any, Literal

class TrainingConfig(BaseModel):
    learning_rate: float
    epochs: int
    device: Literal["cpu", "cuda"]
    
def load_config(raw: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**raw)

print(load_config({"learning_rate": 0.001, "epochs": 10, "device": "cuda"}))
print(load_config({"learning_rate": "3e-4", "epochs": 10, "device": "cuda"}))
print(load_config({"learning_rate": 0.001, "epochs": 10, "device": "GPU"}))

