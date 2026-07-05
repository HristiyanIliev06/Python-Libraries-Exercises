from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Literal, Optional, Annotated

class SearchBound(BaseModel):
    low: float
    high: float

    @model_validator(mode="after")
    def check_bounds(self) -> "SearchBound":
        if self.high <= self.low:
            raise ValueError(f"high must be greater than low, got low={self.low}, high={self.high}")
        return self
    
class HparamSearchConfig(BaseModel):
    experiment_name: str = Field(min_length = 1, max_length = 128, pattern = r'^[a-zA-Z0-9_-]+$')
    search_strategy: Literal["grid", "random", "bayesian"]
    n_trials: int = Field(default = 50, ge = 1, le = 1000)
    learning_rate_bound: SearchBound
    dropout_bound: SearchBound
    batch_sizes: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    output_dir: str = "./results"
    notes: Optional[str] = Field(default = None, max_length = 500)    
    
def load_search_config(data: dict) -> HparamSearchConfig:
    return HparamSearchConfig(**data)

def summarize_search(config: HparamSearchConfig) -> str:
    return f"""Experiment name {config.experiment_name} with strategy {config.search_strategy},
    number of trials {config.n_trials}, lr bound range [{config.learning_rate_bound.low}, {config.learning_rate_bound.high}],
    dropout bound range {config.dropout_bound}, and batch sizes {config.batch_sizes}."""

# 1    
valid_config = {
    "experiment_name": "my_experiment",
    "search_strategy": "grid",
    "n_trials": 100,
    "learning_rate_bound": {"low": 0.0001, "high": 0.01},
    "dropout_bound": {"low": 0.1, "high": 0.5},
    "batch_sizes": [16, 32, 64],
    "output_dir": "./results",
    "notes": "This is a valid hyperparameter search configuration."
}

print(summarize_search(load_search_config(valid_config)))

# 2
try:
    invalid_config1 = {
        "experiment_name": "my_experiment",
        "search_strategy": "grid",
        "n_trials": 100,
        "learning_rate_bound": {"low": 0.1, "high": 0.01}, # Invalid bounds (high <= low)
        "dropout_bound": {"low": 0.1, "high": 0.5},
        "batch_sizes": [16, 32, 64],
        "output_dir": "./results",
        "notes": "This is an invalid hyperparameter search configuration."
    }
    
    print(summarize_search(load_search_config(invalid_config1)))
except ValidationError as e:
    print("Validation error for invalid_config1:", e)
    
# 3
try:
    invalid_config2 = {
        "experiment_name": "my_experiment",
        "search_strategy": "grid",
        "n_trials": 5000, #
        "learning_rate_bound": {"low": 0.0001, "high": 0.01},
        "dropout_bound": {"low": 0.1, "high": 0.5},
        "batch_sizes": [], # Empty list (min_length=1)
        "output_dir": "./results",
        "notes": "This is an invalid hyperparameter search configuration."
    }
    
    print(summarize_search(load_search_config(invalid_config2)))
except ValidationError as e:
    print("Validation error for invalid_config2:", e)