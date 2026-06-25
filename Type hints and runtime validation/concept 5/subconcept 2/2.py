from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class DatasetConfig(BaseModel):
    name: str
    split: Literal["train", "val", "test"]
    max_samples: int
    shuffle: bool = False
    seed: int = 42
    
class PipelineConfig(BaseModel):
    run_name: str
    datasets: list[DatasetConfig] = Field(min_length=1)
    device: Literal["cpu", "cuda"]
    output_dir: str
    debug: bool = False
    
def load_pipeline_config(data: dict) -> PipelineConfig:
    # try:
    #     return PipelineConfig(**data)
    # except ValidationError as e:
    #     raise e
    
    return PipelineConfig(**data)

def summarize_pipeline(config: PipelineConfig) -> str:
    return f"""{', '.join(f"{k} : {v}" for k, v in config if k != "datasets")}, with {len(config.datasets)} datasets where 
        {'. '.join(f"Dataset {dataset.name} has split = {dataset.split} and max_samples = {dataset.max_samples}"
        for dataset in config.datasets)}"""
        
dataset1 = DatasetConfig(name="my_dataset1", split="train", max_samples=1000, shuffle=True)
dataset2 = DatasetConfig(name="my_dataset2", split="val", max_samples=500, shuffle=False)
dataset3 = DatasetConfig(name="my_dataset3", split="test", max_samples=200, shuffle=True)

config_data = {
    "run_name": "my_pipeline",
    "datasets": [dataset1, dataset2, dataset3],
    "device": "cuda",
    "output_dir": "/path/to/output",
    "debug": True
}

pipeline_config = load_pipeline_config(config_data)
print(summarize_pipeline(pipeline_config))

config_data_empty = {
    "run_name": "my_pipeline",
    "datasets": [],
    "device": "tpu",
    "output_dir": "/path/to/output",   
    "debug": True
}

try:
    pipeline_config_empty = load_pipeline_config(config_data_empty)
    print(summarize_pipeline(pipeline_config_empty))
except ValidationError as e:
    print(e)


