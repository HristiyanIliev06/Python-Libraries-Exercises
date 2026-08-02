from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class ModelConfig(BaseModel):
    model_name: str = Field(min_length = 1, max_length = 64, pattern = '^[a-z][a-z0-9_]*$')
    hidden_size: int = Field(ge = 64, le = 4096)
    dropout: float = Field(gt = 0.0, lt = 1.0)
    num_layers: int = Field(default = 6, ge = 1)
    description: Optional[str] = None
    
valid_model = ModelConfig(
    model_name = 'my_model',
    hidden_size = 512,
    dropout = 0.5,
    num_layers = 6,
    description = 'This is a valid model configuration.'
)

try:
    invalid_model = ModelConfig(
        model_name = '2fast', # Invalid model name (does not match the regex pattern)
        hidden_size = 512,
        dropout = 0.5,
        num_layers = 6,
        description = 'This is an invalid model configuration.'
    )
except ValidationError as e:
    print("Validation error for invalid_model:", e)


try:
    model_float = ModelConfig(
    model_name = 'my_model',
    hidden_size = 512,
    dropout = 1.0, # Out of valid value range (0.0, 1.0)
    num_layers = 6,
    description = 'This is a valid model configuration.'
)
except ValidationError as e:
    print("Validation error for invalid_model:", e)