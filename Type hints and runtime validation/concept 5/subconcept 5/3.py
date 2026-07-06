from pydantic import BaseModel, Field, model_validator, field_validator, ValidationError
from typing import Literal, Union, Annotated, Optional

class BatchingConfig(BaseModel):
    max_batch_size: int = Field(ge=1, default=32)
    timeout_ms: int = Field(ge=1, default=100)

class ImagePreprocessConfig(BaseModel):
    resize: tuple[int, int]
    normalize: bool = True
    channels: Literal[1, 3]

class SequenceConfig(BaseModel):
    max_seq_len: int = Field(ge=1, default=512)
    pad_token_id: int = Field(ge=0, default=0)

class TransformerServingConfig(BaseModel):
    model_type: Literal["transformer"]
    model_path: str
    batching: BatchingConfig
    device: Literal["cpu", "cuda"]

class CNNServingConfig(BaseModel):
    model_type: Literal["cnn"]
    model_path: str
    preprocessing: ImagePreprocessConfig
    device: Literal["cpu", "cuda"]

class LSTMServingConfig(BaseModel):
    model_type: Literal["lstm"]
    model_path: str
    sequence: SequenceConfig
    device: Literal["cpu", "cuda"]

Models = Annotated[
    Union[TransformerServingConfig, CNNServingConfig, LSTMServingConfig],
    Field(discriminator="model_type")
]

class ServingPipelineConfig(BaseModel):
    pipeline_name: str = Field(min_length=1, max_length=64)
    models: Annotated[list[Models], Field(min_length=1)]
    log_requests: bool = True
    max_concurrent_requests: int = Field(ge=1, default=10)

    @field_validator("pipeline_name", mode="before")
    @classmethod
    def normalize_pipeline_name(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")

    @model_validator(mode="after")
    def validate_model_paths(self) -> "ServingPipelineConfig":
        seen = set()
        for model in self.models:
            if model.model_path in seen:
                raise ValueError(f"Duplicate model_path: '{model.model_path}'")
            seen.add(model.model_path)
        return self

def load_serving_config(data: dict) -> ServingPipelineConfig:
    return ServingPipelineConfig.model_validate(data)

def summarize_serving(config: ServingPipelineConfig) -> str:
    lines = []
    for model in config.models:
        lines.append(f"{model.model_type} | {model.model_path} | {model.device}")
    return "\n".join(lines)

data = {
    "pipeline_name": "  My Serving Pipeline  ",
    "models": [
        {
            "model_type": "transformer",
            "model_path": "/models/bert",
            "batching": {"max_batch_size": 16, "timeout_ms": 50},
            "device": "cuda"
        },
        {
            "model_type": "cnn",
            "model_path": "/models/resnet",
            "preprocessing": {"resize": [224, 224], "normalize": True, "channels": 3},
            "device": "cpu"
        },
        {
            "model_type": "lstm",
            "model_path": "/models/lstm",
            "sequence": {"max_seq_len": 128, "pad_token_id": 0},
            "device": "cuda"
        }
    ],
    "log_requests": True,
    "max_concurrent_requests": 5
}

config = load_serving_config(data)
print(summarize_serving(config))
print(config.pipeline_name)
print({f"{k} " for k in [model for model in config.models]})