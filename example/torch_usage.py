from typing import Type

import torch
from torch import nn
from torch.nn.functional import one_hot

from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset

from lingxi_mlkit import *

def load_dataset_fn():
    sklearn_iris_dataset = load_iris()
    tensor_dataset = TensorDataset(
        torch.from_numpy(sklearn_iris_dataset.data).float(),
        one_hot(torch.from_numpy(sklearn_iris_dataset.target)).float(),
    )
    return tensor_dataset


class ModelConfig(BaseModelConfig):
    def __init__(
            self,
            activation_func: Type[nn.Module] = nn.ReLU,
            **kwargs
    ):
        super().__init__()
        self.class_num = kwargs.get('class_num', 3)
        self.sequence_layer = kwargs.get('sequence_layer', [4, 16, self.class_num])
        self.activation_func = activation_func


class Model(BaseModel):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__(config)
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

        self.model_layer = []
        for idx in range(len(self.config.sequence_layer) - 1):
            self.model_layer.append(nn.Linear(self.config.sequence_layer[idx], self.config.sequence_layer[idx + 1]))
            if idx <= len(self.config.sequence_layer) - 3:
                self.model_layer.append(self.config.activation_func())
        self.model_layer = nn.Sequential(*self.model_layer)

    def loss(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def forward(self, x):
        return self.model_layer(x)

    def metric(self, x, y_true):
        y_pred = self.forward(x)
        loss = self.loss(y_true, y_pred)
        acc_metric = (torch.argmax(y_true, dim=-1) == torch.argmax(y_pred, dim=-1)).sum() / len(y_pred)
        return loss, {"acc": acc_metric, "loss": loss.item()}

if __name__ == '__main__':
    train_config = BaseTrainConfig()
    train_config.load_dataset_func = load_dataset_fn
    train_config.enable_scheduler = False

    train_dataset = BaseDataset(train_config)

    model_config = BaseModelConfig()


    trainer = BaseTrainer(dataset=train_dataset, train_config=train_config)
    trainer.train(Model, ModelConfig())

