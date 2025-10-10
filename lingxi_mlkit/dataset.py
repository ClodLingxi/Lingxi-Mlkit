from collections.abc import Callable
from typing import Union

import torch
import torch.utils.data as data

from .config import BaseTrainConfig, HintTyping as Ht


class BaseDataset:
    def __init__(self, config=BaseTrainConfig()):
        super(BaseDataset, self).__init__()
        self.config = config
        self.seed_generator = torch.Generator().manual_seed(config.seed)

        self.dataset: data.TensorDataset | data.Dataset = self.load_dataset_from_func(self.config.load_dataset_func)

        self.train_dataset, self.valid_dataset = self.split_train_valid_dataset()
        self.test_dataset = self.get_test_dataset()

        self.train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.valid_dataloader = data.DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        self.test_dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        ) if self.test_dataset else None

    def split_train_valid_dataset(self):
        return data.random_split(
            self.dataset,
            lengths=[1 - self.config.valid_ratio, self.config.valid_ratio],
            generator=self.seed_generator
        )

    def get_test_dataset(self):
        pass

    def get_train_len(self):
        return len(self.dataset) * (1 - self.config.valid_ratio)

    def get_valid_len(self):
        return len(self.dataset) * self.config.valid_ratio

    @staticmethod
    def load_dataset_from_func(
            func: Union[Ht.LoadDataSetType, Ht.PathType, None]=None
        ) -> Union[data.TensorDataset, Callable]:
        if callable(func):
            return func()
        elif func is not None:
            return data.TensorDataset(*torch.load(func))

        raise ValueError("None Loader")