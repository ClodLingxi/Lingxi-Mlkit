import torch

from .config import BaseModelConfig


class BaseModel(torch.nn.Module):
    def __init__(self, config: BaseModelConfig):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError("Forward Not implemented")

    def loss(self, y_true, y_pred):
        raise NotImplementedError("Loss Not implemented")

    def metric(self, x, y_true)-> tuple[torch.Tensor, dict]:
        y_pred = self.forward(x)
        loss = self.loss(y_true, y_pred)
        return loss, {"loss": loss.item()}

    def predict(self, x) -> torch.Tensor:
        y_pred = self.forward(x)
        return y_pred