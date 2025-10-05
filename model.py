import torch

from config import BaseModelConfig


class BaseModel(torch.nn.Module):
    def __init__(self, config: BaseModelConfig):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, batch) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError("Forward Not implemented")

    def loss(self, y_true, y_pred):
        raise NotImplementedError("Loss Not implemented")

    def predict(self, batch) -> tuple[torch.Tensor, dict]:
        loss, metric = self.forward(batch)
        return loss, metric