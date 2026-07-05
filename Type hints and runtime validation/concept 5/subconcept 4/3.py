from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator, Field, ValidationError

class ColumnConfig(BaseModel):
    name: str
    dtype: Literal["int", "float", "str", "bool"]
    nullable: bool = False

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip().replace(" ", "_")

class DataSourceConfig(BaseModel):
    source_name: str = Field(min_length=1, max_length=64)
    format: Literal["csv", "json", "parquet"]
    columns: list[ColumnConfig] = Field(min_length=1)
    target_column: str
    drop_duplicates: bool = True
    sample_fraction: Optional[float] = None

    @field_validator("sample_fraction")
    @classmethod
    def validate_sample_fraction(cls, sf: float) -> float:
        if sf <= 0.0 or sf >= 1.0:
            raise ValueError("sample_fraction must be between 0.0 and 1.0 exclusive")
        return sf

    @model_validator(mode="after")
    def validate_target_column(self) -> "DataSourceConfig":
        column_names = [col.name for col in self.columns]
        if self.target_column not in column_names:
            raise ValueError(f"target_column '{self.target_column}' must match one of: {column_names}")
        return self

def load_source_config(data: dict) -> DataSourceConfig:
    return DataSourceConfig.model_validate(data)

data = {
    "source_name": "sales_data",
    "format": "csv",
    "columns": [
        {"name": " revenue ", "dtype": "float"},
        {"name": "region", "dtype": "str"},
    ],
    "target_column": "revenue",
    "drop_duplicates": True,
    "sample_fraction": 0.8
}

config = load_source_config(data)
print(config.target_column)
print(config.columns[0].name)