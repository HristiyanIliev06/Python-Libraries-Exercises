from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@runtime_checkable
class Serializable(Protocol):
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "Serializable": ...

@dataclass
class ModelConfig:
    lr: float
    batch_size: int

    def to_dict(self) -> dict:
        return {"lr": self.lr, "batch_size": self.batch_size}

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)

config = ModelConfig(lr=0.001, batch_size=32)
print(isinstance(config, Serializable))
print(config.to_dict())

restored = ModelConfig.from_dict({"lr": 0.001, "batch_size": 32})
print(restored.lr)