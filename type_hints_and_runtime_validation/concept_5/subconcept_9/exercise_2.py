from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationError, model_validator


class FeatureStorePayload(TypedDict):
    dataset_name: str
    feature_columns: list[str]
    target_column: str
    split_ratios: dict[Literal["train", "val", "test"], float] 
    metadata: dict[str, str]

class FeatureStoreConfig(BaseModel):
    dataset_name: str
    feature_columns: Annotated[list[str], Field(min_length = 1)]
    target_column: str
    split_ratios: dict[Literal["train", "val", "test"], float] 
    metadata: dict[str, str] = Field(default_factory = dict)

    @model_validator(mode="after")
    def split_ratios_validate(self):
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0")

        if set(self.split_ratios.keys()) != {"train", "val", "test"}:
            raise ValueError("split_ratios must contain exactly 'train', 'val', 'test'")

        return self
    

    @classmethod
    def from_payload(cls, payload: FeatureStorePayload) -> "FeatureStoreConfig":
        return cls.model_validate(payload)

    def primary_features(self) -> list[str]:
        feature_columns = self.feature_columns.copy()
        target_column = self.target_column

        if target_column in feature_columns:
            feature_columns.remove(target_column)

        return feature_columns

    def describe(self) -> str:
        return f"Dataset: {self.dataset_name} | Features: {len(self.feature_columns)} | Split: {'/'.join(str(int(v*100)) for v in self.split_ratios.values())}"

@dataclass
class ProcessedFeatureSet:
    config: FeatureStoreConfig
    feature_matrix_shape: tuple[int, int]
    target_vector_length: int
    processing_time_ms: float
    warnings: list[str] = field(default_factory = list)

def build_feature_pipeline(payload: FeatureStorePayload, matrix_shape: tuple[int, int],
                            target_len: int, processing_time: float) -> ProcessedFeatureSet:

    feature_store_config = FeatureStoreConfig.from_payload(payload)
    warnings: list[str] = []

    for k, v in feature_store_config.split_ratios.items():
        if v < 0.1:
            warnings.append(f"Warning: {k} split below 10%")

    primary_features_count = len(feature_store_config.primary_features())
    if primary_features_count > 20:
        warnings.append(f"Warning: high feature count ({primary_features_count})")

    result = ProcessedFeatureSet(
        config = feature_store_config,
        feature_matrix_shape = matrix_shape,
        target_vector_length = target_len,
        processing_time_ms = processing_time,
        warnings = warnings,
    )

    return result


"""Write three test calls:

Valid payload — 3 features, standard 70/20/10 split
Valid payload that triggers both warnings — tiny val split, 22 features
Invalid payload — split ratios don't sum to 1.0 — catch and print the error
"""

# Test call 1: Valid payload — 3 features, standard 70/20/10 split
valid_payload_1: FeatureStorePayload = {
    "dataset_name": "customer_data",
    "feature_columns": ["age", "income", "purchase_history"],
    "target_column": "churn",
    "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
    "metadata": {"source": "internal"}
}



print("Test Call 1: Valid Payload — 3 features, standard 70/20/10 split")
processed_feature_set_1 = build_feature_pipeline(valid_payload_1, (1000, 3), 1000, 150.0)
print(f"Processed Feature Set 1: {processed_feature_set_1}")

# Test call 2: Valid payload that triggers both warnings — tiny val split, 22 features
valid_payload_2: FeatureStorePayload = {
    "dataset_name": "customer_data_large",
    "feature_columns": [f"feature_{i}" for i in range(22)],
    "target_column": "churn",
    "split_ratios": {"train": 0.85, "val": 0.05, "test": 0.1},
    "metadata": {"source": "internal"}
}

print("\nTest Call 2: Valid Payload that triggers both warnings — tiny val split, 22 features")
processed_feature_set_2 = build_feature_pipeline(valid_payload_2, (10000, 22), 10000, 300.0)
print(f"Processed Feature Set 2: {processed_feature_set_2}")

# Test call 3: Invalid payload — split ratios don't sum to 1.0 — catch and print the error
invalid_payload: FeatureStorePayload = {
    "dataset_name": "customer_data_invalid",
    "feature_columns": ["age", "income", "purchase_history"],
    "target_column": "churn",
    "split_ratios": {"train": 0.6, "val": 0.3, "test": 0.2}, # Sum is 1.1
    "metadata": {"source": "internal"}
}

print("\nTest Call 3: Invalid Payload — split ratios don't sum to 1.0")
try:
    processed_feature_set_3 = build_feature_pipeline(invalid_payload, (1000, 3), 1000, 150.0)
    print(f"Processed Feature Set 3: {processed_feature_set_3}")
except ValidationError as e:
    print(f"Validation Error: {e}")

