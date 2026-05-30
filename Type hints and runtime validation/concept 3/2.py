from typing import Protocol
import dataclasses as dc
from collections.abc import Iterator
import math


class BatchSource(Protocol):
    def __iter__(self) -> list[str]: ...
    def __len__(self) -> int: ...
    @property
    def description(self) -> str: ...


@dc.dataclass
class FileSource:
    batch_size: int
    data: list[str] = dc.field(default_factory=list)

    def __iter__(self) -> Iterator[list[str]]:
        batch = []
        for item in self.data:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    @property
    def description(self) -> str:
        return (
            f"FileSource with batch size {self.batch_size} and {self.__len__()} items"
        )


@dc.dataclass
class MockSource:
    batches: list[list[str]] = dc.field(default_factory=list)

    def __iter__(self) -> Iterator[list[str]]:  # for batch in self.batches:
        yield from self.batches  #     yield batch

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def description(self) -> str:
        return f"MockSource with {len(self.batches)} batches"


def run_pipeline(source: BatchSource) -> None:
    for batch in source:
        print(f"Processing batch: {batch}")

    print(f"Running pipeline with source: {source.description}")


file_source = FileSource(batch_size=4, data=["item1", "item2", "item3", "item4"])
run_pipeline(file_source)

mock_source = MockSource(batches=[["item1", "item2"], ["item3", "item4"]])
run_pipeline(mock_source)
