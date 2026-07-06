from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Literal, Union, Annotated

class BatchingConfig(BaseModel):
    max_batch_size: int = Field(ge = 1, default = 32)
    timeout_ms: int = Field(ge = 1, default = 100)
    
class ImagePreprocessConfig(BaseModel):
    resize: tuple[int, int]
    normalize: bool = True
    channels: Literal[1, 3]
    
class SequenceConfig(BaseModel):
    max_seq_len: int = Field(ge = 1, default = 512)
    pad_token_id: int = Field(ge = 0, default = 0)
    
class TransformerServingConfig(BaseModel):
    model_type: Literal["transformer"]
    model_path: str
    batching: BatchingConfig
    device:  Literal["cpu", "cuda"]
    
class CNNServingConfig(BaseModel):
    model_type: Literal["cnn"]
    model_path: str
    preprocessing: ImagePreprocessConfig
    device:  Literal["cpu", "cuda"]
    
class LSTMServingConfig(BaseModel):
    model_type: Literal["lstm"]
    model_path: str
    sequence: SequenceConfig
    device:  Literal["cpu", "cuda"]

Models = Annotated[Union[TransformerServingConfig, CNNServingConfig, LSTMServingConfig], Field(discriminator = "model_type")]
    
class ServingPipelineConfig(BaseModel):
    pipeline_name: str = Field(min_length = 1, max_length = 64)
    models: Annotated[list[Models], Field(min_length = 1)]
    log_requests: bool = True
    max_concurrent_requests: int = Field(ge = 1, default = 10)
    
    @model_validator(mode="after")
    def validate_model_path(self)->"ServingPipelineConfig":
        seen = set()
        for model in self.models:
            if model.model_path in seen:
                raise ValueError(f"Duplicate model_path: '{model.model_path}'")
        seen.add(model.model_path)
            
        return self
    
def load_serving_config(data: dict) -> ServingPipelineConfig:
    return ServingPipelineConfig(**data)

def summarize_serving(config: ServingPipelineConfig) -> str:
    return "\n".join(f"Model type {model.model_type} with model path {model.model_path} and device {model.device}" for model in config.models)

valid_config = {
    "pipeline_name": "my_serving_pipeline",
    "models": [
        {
            "model_type": "transformer",
            "model_path": "/path/to/transformer_model",
            "batching": {"max_batch_size": 16, "timeout_ms": 50},
            "device": "cuda"
        },
        {
            "model_type": "cnn",
            "model_path": "/path/to/cnn_model",
            "preprocessing": {"resize": (224, 224), "normalize": True, "channels": 3},
            "device": "cpu"
        },
        {
            "model_type": "lstm",
            "model_path": "/path/to/lstm_model",
            "sequence": {"max_seq_len": 128, "pad_token_id": 0},
            "device": "cuda"
        }
    ],
    "log_requests": True,
    "max_concurrent_requests": 5
}

print(summarize_serving(load_serving_config(valid_config)))

try:
    invalid_config = {
        "pipeline_name": "my_serving_pipeline",
        "models": [
            {
                "model_type": "transformer",
                "model_path": "/path/to/transformer_model",
                "batching": {"max_batch_size": 16, "timeout_ms": 50},
                "device": "cuda"
            },
            {
                "model_type": "cnn",
                "model_path": "/path/to/transformer_model",  # Duplicate model path
                "preprocessing": {"resize": (224, 224), "normalize": True, "channels": 3},
                "device": "cpu"
            }
        ],
        "log_requests": True,
        "max_concurrent_requests": 5
    }
    load_serving_config(invalid_config)
except ValidationError as e:
    print(e)
    
try:
    invalid_config2 = {
        "pipeline_name": "my_serving_pipeline",
        "models": [
            {
                "model_type": "transformer",
                "model_path": "/path/to/transformer_model",
                "batching": {"max_batch_size": 16, "timeout_ms": 50},
                "device": "cuda"
            },
            {
                "model_type": "fox",
                "model_path": "/path/to/cnn_model",
                "preprocessing": {"resize": (224, 224), "normalize": True, "channels": 3},
                "device": "cpu"
            }
        ],
        "log_requests": True,
        "max_concurrent_requests": 5
    }
    load_serving_config(invalid_config2)
except ValidationError as e:
    print(e)