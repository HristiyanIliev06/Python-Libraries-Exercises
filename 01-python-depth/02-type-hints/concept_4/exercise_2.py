import dataclasses as dc
from typing import Literal, get_args

Stage = Literal["train", "val", "test"]
Metric = Literal["loss", "accuracy", "f1"]

VALID_STAGES = frozenset(get_args(Stage))
VALID_METRICS = frozenset(get_args(Metric))

@dc.dataclass
class TrainingLogger:
    
    
    logs: dict[Stage, dict[Metric, float]] = dc.field(
        default_factory=lambda: {stage: {} for stage in VALID_STAGES}
    )
    
    def log(self, stage: Stage, metric: Metric, value: float) -> None:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}")
        if metric not in VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric}")
                   
        self.logs[stage][metric] = value
    
    def summary(self, stage: Stage) -> dict[str, float]:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}")
        
        return {metric: value for metric, value in self.logs[stage].items()}
    
tl = TrainingLogger()
tl.log("train", "loss", 0.25)
tl.log("train", "accuracy", 0.8)
print(tl.summary("train"))