class DatasetIterator:
    def __init__(self, samples):
        self.samples = samples
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.samples): #
            raise StopIteration
        sample = self.samples[self.index]
        self.index += 1
        return sample


iterator = DatasetIterator([1, 2, 3, 4, 5])

for i, sample in enumerate(iterator):
    print(sample)
    if i == 2:
        iterator.reset()  # called mid-iteration