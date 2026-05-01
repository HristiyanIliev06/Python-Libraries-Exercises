from typing import Any, Callable, List


class Pipeline:
    def __init__(self, iterator):
        self.iterator = iterator

    def map(self, func):
        return Pipeline(map(func, self.iterator))

    def filter(self, pred: Callable[[Any], bool]):
        return Pipeline(filter(pred, self.iterator))

    def batch(self, n):
        def _batcher(it):
            current = []
            for item in it:
                current.append(item)
                if len(current) == n:
                    yield current
                    current = []
            if current:
                yield current

        return Pipeline(_batcher(self.iterator))

    def __iter__(self):
        return iter(self.iterator)

    def __repr__(self):
        return f"Pipeline({self.iterator})"


pipeline = (
    Pipeline(open("The 5 projects.txt"))
    .map(str.strip)
    .filter(bool)
    .map(str.split)
    .batch(32)
)


for batch in pipeline:
    print(batch)
