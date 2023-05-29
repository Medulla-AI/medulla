import math
from .sampler import Sampler, RandomSampler, SequentialSampler,  BatchSampler


class Dataset:
    """Dataset Base Class.

    All Datasets to be used with DataLoader must have `__len__` which 
    returns the number of datapoints in the dataset and `__getitem__` 
    selects a single datapoint from an index.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


class DataLoader:
    """Combines a Dataset and Sampler for efficiently iterating a Dataset.

    Args:
        dataset (Dataset): Dataset class
        batch_size (int): Batch size; number of data points in a mini-batch
        shuffle (bool): If True, the mini-batches will be randomly sampled.
        drop_last (bool): If True, will ensure every mini-batch is of size `batch_size`
        sampler (Sampler): Sampler class for sampling individual elements
        seed (int): Seed for Jax PRNG
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            drop_last: bool = False,
            sampler: Sampler = None,
            seed: int = None):

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if sampler != None and not isinstance(sampler, Sampler):
            raise ValueError("sampler must be an instance of Sampler")

        if sampler == None:
            sampler = BatchSampler(
                sampler=RandomSampler(dataset, seed=seed) if shuffle
                else SequentialSampler(dataset, seed=seed),
                batch_size=batch_size,
                drop_last=drop_last
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler

        self._iterator = DataLoaderIter(self)

    def __len__(self) -> int:
        length = len(self.dataset)

        if self.drop_last:
            size = length // self.batch_size
        else:
            size = math.ceil(length / self.batch_size)

        return size

    def __iter__(self):
        self._iterator._reset()
        return self._iterator


class DataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._sampler = iter(loader.sampler)
        self._index = 0

    def __next__(self):
        data = self._get_next()
        self._index += 1

        return data

    def _get_next(self):
        index = next(self._sampler)
        data = self._dataset[index]

        return data

    def _reset(self):
        self.index = 0

    def __iter__(self):
        return self
