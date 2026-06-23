from typing import Literal, get_args
from dataclasses import dataclass

Reduction = Literal["mean", "weighted_mean", "sum"]

@dataclass
class LossConfig:
    reduction: Reduction = "mean"
    weight: float = 1.0
    
    def __post_init__(self) -> None:
        if self.reduction not in get_args(Reduction):
            raise ValueError(f"Invalid reduction method: {self.reduction}")

    def apply_reduction(self, values: list[float]) -> float:
        if self.reduction == "mean":
            return sum(values) / len(values)
        if self.reduction == "sum":
            return sum(values)
        if self.reduction == "weighted_mean":
            return sum(v * self.weight for v in values) / len(values)
        
        raise ValueError(f"Unreachable: {self.reduction}")

result = LossConfig(reduction="median").apply_reduction([0.1, 0.3, 0.5])
print(result)