from typing import TypeVar, Generic

T = TypeVar('T', bound = float)

class EmbeddingStore(Generic[T]):
    def __init__(self) -> None:
        self.embeddings: dict[str, T] = {}

    def add(self, id: str, embedding: T) -> None:
        self.embeddings[id] = embedding

    def get(self, id: str) -> T | None:
        return self.embeddings.get(id)

    def get_batch(self, ids: list[str]) -> list[T | None]:
        return [self.get(id) for id in ids]

    def __len__(self) -> int:
        return len(self.embeddings)


if __name__ == "__main__":
    store: EmbeddingStore[list[float]] = EmbeddingStore()
    store.add("1", "True")
    store.add("2", [0.8])

    print(store.get("1"))               # [0.5]
    print(store.get("3"))               # None
    print(store.get_batch(["1", "2", "3", "4"]))  # [[0.5], [0.8], None]
    print(len(store))                   # 2