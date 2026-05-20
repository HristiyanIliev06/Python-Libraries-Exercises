from typing import TypeVar, Generic
from dataclasses import dataclass


T = TypeVar('T', bound = object)

@dataclass
class Scored(Generic[T]):
    item: T
    score: float

def top_k(items: list[Scored[T]], k: int) -> list[T]:
    return [item.item for item in sorted(items, key=lambda x: x.score, reverse=True)[:k]]

if __name__ == "__main__":
    items = [Scored[int](item=1, score=0.5), Scored[int](item=2, score=0.8), Scored[int](item=3, score=0.3)]
    print(top_k(items, 2))  # [2, 1]
    
"""The 'bound=object' in the TypeVar definition allows us to use any type that is a subclass of 'object',
which includes all types in Python.
This means we can use 'Scored' with any type for 'item', such as 'int', 'str',
or even custom classes. In a real ML codebase I'd use more restrictive bounds to ensure that the items have certain properties or methods, but for this example, it allows for maximum flexibility."""