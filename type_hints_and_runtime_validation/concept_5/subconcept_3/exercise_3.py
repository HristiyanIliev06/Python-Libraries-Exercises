from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Literal, Optional, Annotated

class SearchBound(BaseModel):
    low: float
    high: float

    @model_validator(mode="before")
    def check_bounds(self) -> "SearchBound":
        if self["high"] <= self["low"]:
            raise ValueError(f"high must be greater than low") 
        return self

class HparamSearchConfig(BaseModel):
    experiment_name: str = Field(min_length=1, max_length=128)
    search_strategy: Literal["grid", "random", "bayesian"]
    n_trials: int = Field(default=50, gt=1, le=1000)
    learning_rate_bound: SearchBound
    dropout_bound: SearchBound
    batch_sizes: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    output_dir: str = "./results"
    notes: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def check_strategy_trials(self) -> dict:
        if self.search_strategy == "grid" and self.n_trials > 100:
            raise ValueError("grid search should not exceed 100 trials") 
        return self

def load_search_config(data: dict) -> HparamSearchConfig:
    return HparamSearchConfig.model_validate(data) # ?: works fine without model_validate

try: # In case of invalid data, we had better catch the ValidationError and print it
    data = {
        "experiment_name": "exp_01",
        "search_strategy": "random",
        "n_trials": 1, # must be greater than 1
        "learning_rate_bound": {"low": 1e-4, "high": 1e-4},
        "dropout_bound": {"low": 0.1, "high": 0.5},
        "batch_sizes": [16, 32, 64],
        "output_dir": "./results",
    }
    
    config = load_search_config(data)
    print(config)
except ValidationError as e:
    print("Validation error:", e)