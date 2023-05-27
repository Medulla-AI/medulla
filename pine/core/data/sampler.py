import math
from .. import random
from jax.numpy import arange
from data import Dataset


class Sampler:
    """Base class for Samplers.

    All Samplers inheriting from this base class must have the `__iter__` method
    which provides a way to iterate the dataset's indices and the `__len__` method 
    that returns the length of the returned iterators.
    """

    def __init__(self, data, seed: int):
        self.data = data
        self.seed = seed

    def __iter__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)


class SequentialSampler(Sampler):
    """Sequential Sampler.
    Samples elements in the data sequentially.

    Args:
        data (Dataset): Dataset class
        seed (int): Seed for Jax PRNG

    """

    def __init__(self, data: Dataset, seed: int):
        self.see = seed
        self.data = data

    def __iter__(self):
        n = range(len(self.data))
        yield from list(n)


class RandomSampler(Sampler):
    """Random Sampler.
    Samples elements randomly

    Args:
        data (Dataset): Dataset class
        seed (int): Seed for Jax PRNG
    """

    def __init__(self, data: Dataset, seed: int = None):
        self.seed = seed
        self.data = data

    def __iter__(self):
        n = arange(len(self.data))
        yield from random.permutation(self.seed, n).tolist()


class BatchSampler(Sampler):
    """Batch Sampler.
    Samples mini-batches of elements in the dataset sequentially or randomly.

    Args:
        sampler (Sampler): Sampler class for sampling individual elements
        batch_size (int): Number of elements in a mini-batch
        drop_last (bool): If True, will ensure every mini-batch is of size `batch_size`
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []

        for i in self.sampler:
            batch.append(i)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # get left over

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        length = len(self.sampler.data)
        if self.drop_last:
            size = length // self.batch_size
        else:
            size = math.ceil(length / self.batch_size)

        return size
