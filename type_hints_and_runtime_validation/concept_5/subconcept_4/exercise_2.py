from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator, Field, ValidationError

class ColumnConfig(BaseModel):
    name: str
    
    @field_validator("name")
    @classmethod
    def normalize_username(cls, n: str)->str:
        return n.strip().replace(" ", "_")
    
    dtype: Literal["int", "float", "str", "bool"]
    nullable: bool = False
    alias: Optional[str] = None
    
class DataSourceConfig(BaseModel):
    source_name: str = Field(min_length = 1, max_length = 64)
    format: Literal["csv", "json", "parquet"]
    columns: list[ColumnConfig] = Field(min_length = 1)
    target_column: str
    drop_duplicates: bool = True
    sample_fraction: Optional[float] = None
    
    @field_validator("sample_fraction")
    @classmethod
    def validate_sample_fraction(cls, sf: float):
        if (sf<=0.0 or sf>=1.0):
            raise ValueError("Sample_fraction must be between 0.0 and 1.0 exclusive!")
        
        return sf
    
    @model_validator(mode="after")
    def validate_columns(self) -> "DataSourceConfig":
        if self.target_column not in [col.name for col in self.columns]:
            raise ValueError("The targetted column must be contained in the columns field!")
        
        return self
    
def load_source_config(data: dict) -> DataSourceConfig:
    return DataSourceConfig(**data)

def summarize_source(config: DataSourceConfig) -> str:
    summary = f"Source Name: {len(config.columns)}\n"
    summary += f"Format: {config.format}\n"
    summary += f"Columns: {[col.name for col in config.columns]}\n"
    summary += f"Target Column: {config.target_column}\n"
    summary += f"Drop Duplicates: {config.drop_duplicates}\n"
    if config.sample_fraction is not None:
        summary += f"Sample Fraction: {config.sample_fraction}\n"
    return summary

valid_config_data = {
    "source_name": "my_data_source",
    "format": "csv",
    "columns": [
        {"name": "id", "dtype": "int"},
        {"name": "value", "dtype": "float"},
        {"name": "category", "dtype": "str", "nullable": True}
    ],
    "target_column": "value",
    "drop_duplicates": True,
    "sample_fraction": 0.5
}

print(summarize_source(load_source_config(valid_config_data)))

try:
    invalid_config_data = {
        "source_name": "my_data_source",
        "format": "csv",
        "columns": [
            {"name": "id", "dtype": "int"},
            {"name": "value", "dtype": "float"},
            {"name": "category", "dtype": "str", "nullable": True}
        ],
        "target_column": "string_not_in_columns",
        "drop_duplicates": True,
        "sample_fraction": 0.5
    }

    print(summarize_source(load_source_config(invalid_config_data)))
except ValidationError as e:
    print(e)
    
try:
    invalid_config_data = {
        "source_name": "my_data_source",
        "format": "csv",
        "columns": [
            {"name": "id", "dtype": "int"},
            {"name": "value", "dtype": "float"},
            {"name": "category", "dtype": "str", "nullable": True}
        ],
        "target_column": "value",
        "drop_duplicates": True,
        "sample_fraction": 1.0
    }

    print(summarize_source(load_source_config(invalid_config_data)))
except ValidationError as e:
    print(e)

