from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Literal, Optional

class DeviceConfig(BaseModel):
    device_type: Literal["cpu", "cuda", "mps"]
    num_gpus: Optional[int] = Field(default=None, gt=0)

class LoRAConfig(BaseModel):
    rank: int = Field(gt=0)
    alpha: float = Field(gt=0.0)
    target_modules: list[str] = Field(min_length=1)

class FinetuneConfig(BaseModel):
    model_name: str = Field(min_length=1)
    device: DeviceConfig
    lora: Optional[LoRAConfig] = None
    learning_rate: float = Field(gt=0.0, lt=1.0)
    use_lora: bool = True

    @field_validator("model_name")
    @classmethod # was missing
    def validate_model_name(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError("model_name must be in 'org/model' format")
        return v

    @model_validator(mode="after")
    def validate_lora_consistency(self) -> "FinetuneConfig":
        if self.use_lora and self.lora is None:
            raise ValueError("use_lora is True but no LoRAConfig provided")
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_device_gpu_consistency(cls, values: dict) -> dict:
        device = values.get("device", {})
        device_type = device.get("device_type")
        num_gpus = device.get("num_gpus")
        if device_type == "cpu" and num_gpus is not None:
            raise ValueError("num_gpus should not be set when device_type is cpu")
        return values
    # @model_validator(mode="after") # @classmethod is not needed for model_validator
    # def validate_device_gpu_consistency(self) -> "FinetuneConfig":
    #     device = self.device
    #     device_type = device.device_type
    #     num_gpus = device.num_gpus
    #     if device_type == "cpu" and num_gpus is not None:
    #         raise ValueError("num_gpus should not be set when device_type is cpu")
    #     return self
    
valid_data = {
    "model_name": "my_org/my_model",
    "device": {
        "device_type": "cuda",
        "num_gpus": 2
    },
    "lora": {
        "rank": 4,
        "alpha": 16.0,
        "target_modules": ["module1", "module2"]
    },
    "learning_rate": 0.001,
    "use_lora": True
}

invalid_data = {
    "model_name": "my/model",  # Invalid format
    "device": {
        "device_type": "cpu",
        "num_gpus": 1  # Invalid: num_gpus should not be set for CPU
    },
    "lora": None,  # Invalid: use_lora is True but no LoRAConfig provided
    # "lora": {
    #     "rank": 4,
    #     "alpha": 16.0,
    #     "target_modules": ["module1", "module2"]
    # },
    "learning_rate": 0.001,
    "use_lora": True
}

try:
    config = FinetuneConfig.model_validate(valid_data)
    print("Valid configuration loaded successfully:")
    print(config)
except ValidationError as e:
    print("Validation error for valid_data:", e)
    
try:
    config = FinetuneConfig.model_validate(invalid_data)
    print("Invalid configuration loaded successfully:")
    print(config)
except ValidationError as e:
    print("Validation error for invalid_data:", e)
    