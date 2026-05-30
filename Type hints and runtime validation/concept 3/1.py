from typing import Protocol
from dataclasses import dataclass


class Describable(Protocol):
    def describe(self) -> str: ...


def print_description(item: Describable) -> None:
    print(item.describe())


@dataclass
class ModelCard:
    name: str
    version: str

    def describe(self) -> str:
        return f"Model: {self.name} v{self.version}"


@dataclass
class DatasetInfo:
    source: str
    size: int

    def describe(self) -> str:
        return f"Dataset: {self.source} ({self.size} samples)"


mc = ModelCard(name="Sonnet 4.6", version="1.0")
print_description(mc)

dsi = DatasetInfo(source="Anthropic", size=1_000_000)
print_description(dsi)