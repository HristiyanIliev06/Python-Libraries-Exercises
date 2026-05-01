class TokenBatcher:
    def __init__(self, filepath, batch_size):
        self.filepath = filepath
        self.batch_size = batch_size

    def _generate(self):
        with open(self.filepath) as f:
            batch = []
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                batch.extend(tokens)
                if len(batch) >= self.batch_size:
                    yield batch[: self.batch_size]
                    batch = batch[self.batch_size :]
            if batch:
                yield batch

    def __iter__(self):
        return self._generate()


token_batcher = TokenBatcher("The 5 projects.txt", 5)

# print(list(token_batcher))

for token in token_batcher:
    print(token)
