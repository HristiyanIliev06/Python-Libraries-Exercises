

import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
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
        try:
            if overrides is not None:
                for k, v in overrides.items():
                    k: str
                    if v is not None:
                        os.environ[k.upper()] = str(v)
                        overrides_keys.append(k)

            return PipelineSettings()
        finally:
            for k in overrides_keys:
                os.environ.pop(k)

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

    