from dataclasses import dataclass, field
import os
import string
from typing import Literal
import uuid
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    configure = SettingsConfigDict(env_file="pipeline.env", env_file_encoding="utf-8", case_sensitive=False)
    dataset_path: str
    output_dir: str
    batch_size: int = Field(default = 32, ge = 1, le = 512) 
    max_token_length: int = Field(default = 512, ge = 64, le = 4096)
    environment: Literal["dev", "staging", "production"]
    enable_augmentation: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode = "after")
    def validate_enable_augmentation(self):
        if self.enable_augmentation and self.environment == "production":
            raise ValueError("Augmentation must be disabled in production")

        return self

    def load_pipeline_settings(overrides: dict | None = None) -> "PipelineSettings":
        overrides_keys = []
        if overrides is not None:
            for k, v in overrides.items():
                k: str
                if v is not None:
                    os.environ[k.upper()] = str(v)
                    overrides_keys.append(k.upper())
        try:
            return PipelineSettings()
        finally:
            for k in overrides_keys:
                os.environ.pop(k, None)

class RawTextSample(BaseModel):
    sample_id: str = Field(pattern = r"^[a-z0-9\-]+$") 
    text: str
    label: str
    source: Literal["web", "books", "academic", "synthetic"]
    metadata: dict[str, str] = Field(default_factory = dict)
        
    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v: str):
        if not v.strip():
            raise ValueError("Text must be non-empty")

        return v

class PreprocessingConfig(BaseModel):
    lowercase: bool = True
    strip_punctuation: bool = False
    max_length: int | None = Field(default = None, gt = 0)
    filters: list[Literal["profanity", "pii", "boilerplate"]] = Field(default_factory = list)

    @model_validator(mode="after")
    def validate_max_length(self, info: ValidationInfo):
        settings: PipelineSettings = info.context.get("settings") if info.context else None
        if settings and self.max_length is not None:
            if self.max_length > settings.max_token_length:
                raise ValueError(
                f"max_length {self.max_length} exceeds "
                f"settings.max_token_length {settings.max_token_length}"
            )
        return self
    
@dataclass
class ProcessedSample:
    sample_id: str
    original_text: str
    processed_text: str
    token_count: int
    was_truncated: bool
    applied_filters: list[str] = field(default_factory = list)

@dataclass
class PipelineBatch:
    batch_id: str
    samples: list[ProcessedSample]
    settings_snapshot: PipelineSettings
    warnings: list[str] = field(default_factory = list)

    def size(self) -> int:
        return len(self.samples)

    def truncated_count(self) -> int:
        return len([item for item in self.samples if item.was_truncated])

    def describe(self) -> str: 
        return f"Batch {self.batch_id}: {self.size()} samples | {self.truncated_count()} truncated | warnings: {self.warnings}"

def run_pipeline(raw_samples: list[dict], preprocessing_config: dict, settings: PipelineSettings) -> PipelineBatch:
    validated_samples: list[RawTextSample] = []
    warnings = []

    for raw_sample in raw_samples:
        try:
            validated_sample = RawTextSample.model_validate(raw_sample)
            validated_samples.append(validated_sample)
        except ValidationError as error:
            warnings.append(f"Skipped invalid sample {raw_sample.sample_id}: {error}")

    config = PreprocessingConfig.model_validate(preprocessing_config, context={"settings": settings})

    processed_samples = []
    for sample in validated_samples:
        text = sample.text  # already stripped of whitespace by field_validator

        if config.lowercase:
            text = text.lower()

        if config.strip_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = text.split()
        was_truncated = False
        if config.max_length is not None and len(tokens) > config.max_length:
            tokens = tokens[:config.max_length]
            was_truncated = True

        processed_text = " ".join(tokens)
        token_count = len(tokens)
        applied_filters = list(config.filters)

        processed_samples.append(ProcessedSample
            (
                sample_id = sample.sample_id,
                original_text = sample.text,
                processed_text = processed_text,
                token_count = token_count,
                was_truncated = was_truncated,
                applied_filters = applied_filters
            )
        )

    batch_id = f"batch-{uuid.uuid4().hex[:8]}"

    return PipelineBatch(
        batch_id = batch_id,
        processed_samples = processed_samples,
        settings_snapshot = settings,
        warnings = warnings,
    )

"""Test calls
Valid batch — 3 samples, lowercase + truncation at 10 tokens, dev environment
Batch with one invalid sample (bad sample_id pattern) — verify it's skipped and warning is present
PreprocessingConfig where max_length exceeds settings.max_token_length — catch and print ValidationError
Production settings with enable_augmentation=True — catch and print ValidationError from load_pipeline_settings"""

valid_raw_samples = [
    {"sample_id": "sample-001", "text": "This is a sample text for testing.", "label": "test", "source": "web"},
    {"sample_id": "sample-002", "text": "Another example of text data.", "label": "example", "source": "books"},
    {"sample_id": "sample-003", "text": "Yet another sample text to process.", "label": "sample", "source": "academic"},
]

valid_batch_settings = PipelineSettings(
    dataset_path = "/path/to/dataset",
    output_dir = "/path/to/output",
    batch_size = 32,
    max_token_length = 512,
    environment = "dev",
    enable_augmentation = False,
    log_level = "INFO"
)

valid_batch = run_pipeline(
    raw_samples = valid_raw_samples,
    preprocessing_config = {"lowercase": True, "strip_punctuation": False, "max_length": 10, "filters": []},
    settings = valid_batch_settings,
)
print(valid_batch.describe())