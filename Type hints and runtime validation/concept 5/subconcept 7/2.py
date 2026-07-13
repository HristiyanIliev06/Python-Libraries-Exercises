from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Any, Literal, Optional
#import math
import json
from pathlib import Path
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

def save_config(config: ExperimentConfig, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    json_string = config.model_dump_json()
    try:
        with open(p, "w") as f:
            f.write(json_string)
    except OSError as e:
        raise IOError(f"Failed to write config to {p}: {e}")
    
def load_config(path: str) -> ExperimentConfig:
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    return ExperimentConfig.model_validate(raw)

def diff_configs(a: ExperimentConfig, b: ExperimentConfig) -> dict:
    a_dict = a.model_dump()
    b_dict = b.model_dump()
    diff_dict: dict[str, Any] = {}
    
    for key in a_dict:
        if a_dict[key] != b_dict[key]:
            v_a = a_dict[key]
            v_b = b_dict[key]
            diff_dict[key] = {"a": v_a, "b": v_b}
            
    return diff_dict

def summarize_config(config: ExperimentConfig) -> dict:
    return config.model_dump(exclude_defaults = True, exclude_none = True)

a_config = ExperimentConfig(
    experiment_name = "test_experiment",
    device = "cuda",
    seed = 42,
    data = DataConfig(
        train_path = "data/train.csv",
        val_path = "data/val.csv",
        max_samples = 1000,
        num_workers = 4
    ),
    model = ModelConfig(
        architecture = "transformer",
        hidden_dim = 256,
        num_layers = 4,
        dropout = 0.1
    ),
    optimizer = OptimizerConfig(
        name = "adam",
        lr = 0.001,
        weight_decay = 0.01
    ),
    scheduler = SchedulerConfig(
        name = "cosine",
        max_steps = 10000,
        warmup_steps = 1000
    )
)

b_config = ExperimentConfig(
    experiment_name = "test_experiment",
    device = "cuda",
    seed = 42,
    data = DataConfig(
        train_path = "data/train.csv",
        val_path = "data/val.csv",
        max_samples = 1000,
        num_workers = 4
    ),
    model = ModelConfig(
        architecture = "transformer",
        hidden_dim = 512,
        num_layers = 4,
        dropout = 0.1
    ),
    optimizer = OptimizerConfig(
        name = "adam",
        lr = 0.001,
        weight_decay = 0.01
    ),
    scheduler = SchedulerConfig(
        name = "cosine",
        max_steps = 10000,
        warmup_steps = 1000
    )
)

try:
    save_config(a_config, "Type hints and runtime validation/concept 5/subconcept 7/config.json")
    print("Saved config to config.json")
except OSError  as e:
    print("Error saving config:", e)

print("Loading config from config.json")
try:
    loaded_config = load_config("config.json")
    print("Loaded config:", loaded_config)
except FileNotFoundError as e:
    print(e)


print("Diff between a_config and b_config:")
diff = diff_configs(a_config, b_config)
print(diff)

print("\n=====================================================================\n")

print(summarize_config(a_config))
