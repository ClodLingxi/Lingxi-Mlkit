from typing import Union

import torch
import torch.utils.data as data

from .config import BaseTrainConfig, HintTyping as Ht


class BaseDataset:
    def __init__(self, config=BaseTrainConfig()):
        super(BaseDataset, self).__init__()
        self.config = config
        self.seed_generator = torch.Generator().manual_seed(config.seed)

        self.dataset: dict[str, data.TensorDataset] = self.load_dataset_from_func(self.config.load_dataset_func)

        self.train_dataset, self.valid_dataset = self.get_train_valid_dataset()
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

    def get_train_valid_dataset(self):
        dataset = self.dataset['train']
        return data.random_split(
            dataset,
            lengths=[1 - self.config.valid_ratio, self.config.valid_ratio],
            generator=self.seed_generator
        )

    def get_test_dataset(self):
        return self.dataset.get("test", None)

    def get_train_len(self):
        return len(self.dataset['train']) * (1 - self.config.valid_ratio)

    def get_valid_len(self):
        return len(self.dataset['train']) * self.config.valid_ratio

    @staticmethod
    def load_dataset_from_func(
            func: Union[Ht.LoadDataSetType, Ht.PathType, None]=None
        ) -> dict[str, data.TensorDataset]:
        if func is None:
            return {
                "train": data.TensorDataset(),
                "test": None
            }

        return func()