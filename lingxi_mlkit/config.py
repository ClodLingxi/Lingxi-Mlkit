from os import PathLike
from typing import Type, Callable, Union, TypedDict, Optional, Any

import numpy as np
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR
from torch.utils.data import Dataset


class HintTyping:
    PathType = Union[str, bytes, PathLike]
    class DatasetTye(TypedDict, total=False):
        train: Dataset[Any]
        test: Optional[Dataset[Any]]
        valid: Optional[Dataset[Any]]


class BaseTrainConfig:
    def __init__(self, device="cuda"):
        self.batch_size = 8
        self.epochs = 10

        self.learning_rate = 1e-3
        self.weight_decay = 1e-3
        self.optimizer: Type[Optimizer] = AdamW
        self.optimizer_params = {"lr": self.learning_rate, "weight_decay": self.weight_decay}

        self.warmup_epochs = 1

        self.device = device
        self.seed = 666

        self.small_train_ratio = 1
        self.valid_ratio = 0.1

        self.load_state_dict_path = None
        self.load_dataset_func: Callable[[], HintTyping.DatasetTye] | None = None

        self.train_metric: dict["str", Union[dict["str", Union[Callable, None]], None]] = {
            "loss": {
                "mean": lambda x: np.mean(x),
            }
        }

        self.enable_scheduler = True
        self.enable_swanlab = True
        self.enable_tqdm = True

        self.print_local = False

    @staticmethod
    def get_scheduler(optimizer: Optimizer, num_warmup_step: int, max_step: int, min_lr = 1e-5):
        def warmup_lambda(epoch: int):
            if num_warmup_step == 0:
                return 1.0
            return float(epoch + 1) / float(num_warmup_step)

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_step - num_warmup_step,
            eta_min=min_lr
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_step]
        )

        return scheduler


class BaseModelConfig:
    def __init__(self):
        super().__init__()
        pass
