from typing import Union, Callable, Sized, Any

import torch
import torch.utils.data as data

from .config import BaseTrainConfig, HintTyping as Ht


class BaseDataset:
    def __init__(self, config=BaseTrainConfig()):
        super(BaseDataset, self).__init__()
        self.config = config
        self.seed_generator = torch.Generator().manual_seed(config.seed)

        self.dataset: Ht.DatasetTye = self.load_dataset_from_func(self.config.load_dataset_func)

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

        if self.dataset["valid"] is not None:
            valid_dataset: data.Dataset[Any] = self.dataset["valid"]
            return dataset, valid_dataset

        train, valid = data.random_split(
            dataset,
            lengths=[1 - self.config.valid_ratio, self.config.valid_ratio],
            generator=self.seed_generator
        )

        return train.dataset, valid.dataset

    def get_test_dataset(self):
        return self.dataset['test']

    def get_train_len(self):
        dataset = self.dataset['train']
        if not isinstance(dataset, Sized):
            raise TypeError("Dataset must be of type Sized")

        return int(len(dataset) * (1 - self.config.valid_ratio))

    def get_valid_len(self):
        dataset = self.dataset['train']
        if not isinstance(dataset, Sized):
            raise TypeError("Dataset must be of type Sized")

        return int(len(dataset) * self.config.valid_ratio)

    @staticmethod
    def load_dataset_from_func(
            func: Union[Callable[[], Ht.DatasetTye], None]=None
        ) -> Ht.DatasetTye:
        if func is None:
            return {
                "train": data.TensorDataset(),
                "valid": data.TensorDataset(),
            }

        return func()