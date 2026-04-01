def token_batches(filepath, batch_size):
    with open(filepath) as f:
        batch = []
        for line in f:
            tokens = line.strip().split(" ")
            if not tokens:
                continue
            batch.extend(tokens)
            if len(batch) >= batch_size:
                yield batch[:batch_size]
                batch = batch[batch_size:]
        if batch:
            yield batch


print(list(token_batches("The 5 projects.txt", 5)))
