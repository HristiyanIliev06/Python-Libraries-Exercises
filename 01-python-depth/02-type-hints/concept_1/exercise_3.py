from typing import TypeVar, Generic
from collections.abc import Callable

T = TypeVar("T")
S = TypeVar("S")

class Pipeline(Generic[T]):
    def __init__(self, data: list[T]):
        self.data = data
    
    def transform(self, fn: Callable[[T], S]) -> "Pipeline[S]":
        return Pipeline([fn(item) for item in self.data])
    
    def first(self) -> T | str:
        return self.data[0] if self.data else "Empty Pipeline"

pipe: Pipeline[int] = Pipeline([1, 2, 3])
result = pipe.transform(str).first()
print(result)  