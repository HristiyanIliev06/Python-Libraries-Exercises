from pydantic import BaseModel, Field
from typing import Literal, Optional

class DatasetConfig(BaseModel):
    name: str
    split: Literal["train", "val", "test"]
    max_samples: int = Field(gt=0)
    shuffle: bool = False

class PipelineConfig(BaseModel):
    run_name: str
    datasets: list[DatasetConfig] = Field(min_length=1)
    device: Literal["cpu", "cuda"]
    output_dir: Optional[str] = None # requires default value
    debug: bool = False

def load_pipeline_config(data: dict) -> PipelineConfig:
    return PipelineConfig.model_validate(data)

def summarize_pipeline(config: PipelineConfig) -> str:
    lines = [f"Run: {config.run_name}", f"Device: {config.device}"]
    for ds in config.datasets:
        lines.append(f"  - {ds.name} ({ds.split}): {ds.max_samples} samples")
    return "\n".join(lines) # wrong symbol; no new paragraphs between list items

data = {
    "run_name": "experiment_01",
    "datasets": [
        {"name": "train_set", "split": "train", "max_samples": 1000},
        {"name": "val_set", "split": "val", "max_samples": 50}, # max_samples should be non-negative; raises ValidationError
    ],
    "device": "cuda",
    "debug": False
}

config = load_pipeline_config(data)
print(summarize_pipeline(config))