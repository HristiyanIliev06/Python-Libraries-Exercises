from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T', bound=object)

@dataclass
class Scored(Generic[T]):
    item: T
    score: float

def top_k(items: list[Scored[T]], k: int) -> list[Scored[T]]:
    return sorted(items, key=lambda x: x.score, reverse=True)[:k]

if __name__ == "__main__":
    items = [
        Scored(item=1, score=0.5),
        Scored(item=2, score=0.8),
        Scored(item=3, score=0.3),
    ]
    results = top_k(items, 2)
    for r in results:
        print(r.item, r.score)  # 2 0.8 / 1 0.5