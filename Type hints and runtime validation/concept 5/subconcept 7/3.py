from pydantic import BaseModel, Field, model_validator
from typing import Optional
import json
from pathlib import Path

class AugmentationConfig(BaseModel):
    flip: bool = True
    rotation_degrees: float = Field(default=0.0, ge=0.0, le=360.0)
    cutout_prob: float = Field(default=0.0, ge=0.0, le=1.0)

class PipelineConfig(BaseModel):
    name: str
    augmentation: Optional[AugmentationConfig] = None
    batch_size: int = Field(gt=0)
    cache_dir: Optional[str] = None
    max_cache_gb: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_cache_consistency(self) -> "PipelineConfig":
        if self.cache_dir is not None and self.max_cache_gb is None:
            raise ValueError("max_cache_gb must be set if cache_dir is provided")
        return self

def load_pipeline_config(path: str) -> PipelineConfig:
    if not Path(path).exists(): # Missing pathlib module import
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return PipelineConfig.model_validate(json.load(f))

def export_pipeline_config(config: PipelineConfig) -> str:
    return json.dumps(config.model_dump(exclude_none=True))

def restore_pipeline_config(raw: str) -> PipelineConfig:
    data = json.loads(raw)
    return PipelineConfig.model_validate(data)
# **data bypasses the validation entry point. If data contained nested dicts (like augmentation),
# Pydantic still coerces them — but it's not idiomatic and breaks if you add model_validate hooks later

valid = {
    "name": "training_pipeline",
    "augmentation": {
        "flip": True,
        "rotation_degrees": 45.0,
        "cutout_prob": 0.2
    },
    "batch_size": 32,
    "cache_dir": "/tmp/cache",
    "max_cache_gb": 10.0
}

config = PipelineConfig.model_validate(valid)
exported = export_pipeline_config(config)
print("Exported config:", exported)
restored = restore_pipeline_config(exported)
print("Restored config:", restored)
assert config == restored

valid_no_aug = {
    "name": "training_pipeline",
    "batch_size": 32,
    "cache_dir": "/tmp/cache",
    "max_cache_gb": 10.0
}

config = PipelineConfig.model_validate(valid_no_aug)
exported = export_pipeline_config(config)
restored = restore_pipeline_config(exported)
assert config == restored