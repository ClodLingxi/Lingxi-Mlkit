from pathlib import Path
import os
import random
from typing import Type

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm
import swanlab

from config import BaseTrainConfig, BaseModelConfig
from dataset import BaseDataset
from model import BaseModel

print_cuda = lambda msg, device="cuda": print(msg, round(torch.cuda.memory_allocated(device) / (1024 ** 3), 2), "GB")

class BaseTrainer:
    def __init__(self, dataset: BaseDataset, train_config: BaseTrainConfig):
        self.train_config = train_config
        self.dataset = dataset
        self.set_seed(train_config.seed)

        self.model = None
        self.optimizer = None
        self.scheduler = None

    def train(
            self,
            model: Type[BaseModel] = None, model_config=BaseModelConfig(),
            project_name="ExpProject", experiment_name="BaseExp",
    ):
        swanlab.init(
            project_name=project_name,
            experiment_name=experiment_name,
            config=self.train_config.__dict__ | model_config.__dict__
        )

        self.model = model(model_config)

        print(self.model)

        self.optimizer = self.train_config.optimizer(self.model.parameters(), lr=self.train_config.learning_rate)
        self.scheduler = self.train_config.get_scheduler(
            optimizer=self.optimizer,
            num_warmup_step=self.dataset.get_train_len() * self.train_config.warmup_epochs,
            max_step=self.dataset.get_train_len() * self.train_config.epochs
        ) if self.train_config.enable_scheduler else None

        self.load_state_dict(self.train_config.load_state_dict_path)

        self.epoch_train()

        swanlab.finish()


    def epoch_train(self):
        train_loader = self.dataset.train_dataloader
        valid_loader = self.dataset.valid_dataloader

        for epoch in range(self.train_config.epochs):
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
                self.model.train()

                loss, metric = self.model(batch)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.swanlab_log(metric | {"loss": loss.item()}, tag="train")

            for batch in tqdm(valid_loader, desc=f"Validating Epoch {epoch}"):
                self.model.eval()
                with torch.no_grad():
                    loss, metric = self.model.predict(batch)
                    self.swanlab_log(metric | {"loss": loss.item()}, tag="valid")



    def load_state_dict(self, state_dict_path: Path | None):
        if state_dict_path is None:
            return
        checkpoint = torch.load(state_dict_path, map_location=self.train_config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    @staticmethod
    def swanlab_log(log_dict: dict, tag=None, **kwargs):
        if tag is not None:
            new_log_dict = {k + "/" + tag: v for k, v in log_dict.items()}
            log_dict = new_log_dict
        swanlab.log(data=log_dict, **kwargs)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)