from collections.abc import Callable
from pathlib import Path
import os
import random
from typing import Type

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from tqdm import tqdm
import swanlab

from .config import BaseTrainConfig, BaseModelConfig
from .dataset import BaseDataset
from .model import BaseModel

print_cuda = lambda msg, device="cuda": print(msg, round(torch.cuda.memory_allocated(device) / (1024 ** 3), 2), "GB")

class BaseTrainer:
    def __init__(self, dataset: BaseDataset, train_config: BaseTrainConfig):
        self.train_config = train_config
        self.dataset = dataset
        self._set_seed(train_config.seed)

        self.model: BaseModel = None
        self.optimizer = None
        self.scheduler = None

    def _init_trainer(self, model, model_config):
        self.model = model(model_config)
        self.optimizer = self.train_config.optimizer(self.model.parameters(), lr=self.train_config.learning_rate)
        self.scheduler = self.train_config.get_scheduler(
            optimizer=self.optimizer,
            num_warmup_step=self.dataset.get_train_len() * self.train_config.warmup_epochs,
            max_step=self.dataset.get_train_len() * self.train_config.epochs
        ) if self.train_config.enable_scheduler else None
        self.load_state_dict(self.train_config.load_state_dict_path)

    def train(
            self,
            model: Type[BaseModel] = None, model_config=BaseModelConfig(),
            project_name="ExpProject", experiment_name="BaseExp",
    ):
        if self.train_config.enable_swanlab:
            swanlab.init(
                project_name=project_name,
                experiment_name=experiment_name,
                config=self.train_config.__dict__ | model_config.__dict__
            )

        self._init_trainer(model, model_config)
        print(self.model)

        for epoch in range(self.train_config.epochs):
            self.one_epoch(self.dataset.train_dataloader, no_grad=False, epoch=epoch)
            self.one_epoch(self.dataset.valid_dataloader, no_grad=True, epoch=epoch)

        if self.train_config.enable_swanlab:
            swanlab.finish()


    def test(self):
        test_loader = self.dataset.test_dataloader

        if self.model is None:
            raise RuntimeError("Model Not Loaded")

        if test_loader is None:
            Warning("Test Dataloader was None")
            return

        y_pred_result: list[torch.Tensor] = []

        for x in tqdm(test_loader, desc=f"Test Inference"):
            self.model.eval()
            y_pred = self.model.predict(x)
            y_pred_result.append(y_pred)

        self.test_print(y_pred_result)

    def one_epoch(self, dataloader, no_grad=False, epoch=-1):
        torch.set_grad_enabled(not no_grad)
        tag = "Train" if not no_grad else "Valid"

        epoch_metric = {}

        for batch in tqdm(
                dataloader,
                desc=f"{tag} Epoch {epoch}",
                disable=not self.train_config.enable_tqdm
        ):
            if no_grad:
                self.model.eval()
            else:
                self.model.train()


            x, y = batch
            loss, metric = self.model.metric(x=x, y_true=y)

            if not no_grad:
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            self.swanlab_log(metric, tag=tag)

            for metric_name, metric in metric.items():
                if "!epoch!" in metric_name:
                    metric_name = metric_name.replace("!epoch!", "")

                if metric_name not in self.train_config.train_metric.keys():
                    continue

                if metric_name not in epoch_metric.keys():
                    epoch_metric[metric_name] = []
                epoch_metric[metric_name].append(metric)

        for metric_name, handle_func in self.train_config.train_metric.items():
            self.swanlab_log({metric_name: epoch_metric[metric_name]}, tag=tag, handle_func=handle_func)


    def load_state_dict(self, state_dict_path: Path | None):
        if state_dict_path is None:
            return
        checkpoint = torch.load(state_dict_path, map_location=self.train_config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def test_print(self, test_result):
        raise RuntimeWarning("test print function is not implemented")


    def swanlab_log(self, log_dict: dict, tag=None, handle_func: dict[str, Callable]=None, **kwargs):
        if not self.train_config.enable_swanlab:
            return



        if handle_func is not None:
            for func_k, func in handle_func.items():
                log_dict = {k + "_" + func_k: func(v) for k, v in log_dict.items()}

        if tag is not None:
            new_log_dict = {}
            for k, v in log_dict.items():
                if "!epoch!" in k:
                    continue
                new_log_dict[k + "/" + tag] = v
            log_dict = new_log_dict

        swanlab.log(data=log_dict, print_to_console=self.train_config.print_local, **kwargs)

    @staticmethod
    def _set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)