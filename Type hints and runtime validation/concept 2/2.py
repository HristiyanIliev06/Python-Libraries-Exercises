import dataclasses as dc
from uuid import uuid4

@dc.dataclass(frozen = True)
class TrainingConfig:
    model_name: str
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_steps: int = 10_000
    device: str = "cpu"
    tags: list[str] = dc.field(default_factory = list)
    run_id: str = dc.field(init = False)
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", f"{self.model_name}-{uuid4().hex[:8]}")   
    
    
config = TrainingConfig(model_name = "gpt-3")
print(config.run_id)