import math

import attr
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict
from torch.utils.data import DataLoader

import utils

@attr.s(auto_attribs=True)
class LinearClassifierMethodParams:
    # encoder model selection
    encoder_arch: str = "resnet18"
    embedding_dim: int = 512

    # data-related parameters
    dataset_name: str = "stl10"
    batch_size: int = 256

    # optimization parameters
    lr: float = 30.0
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_epochs: int = 100

    # data loader parameters
    num_data_workers: int = 4
    drop_last_batch: bool = True
    pin_data_memory: bool = True
    multi_gpu_training: bool = False


class LinearClassifierMethod(pl.LightningModule):
    model: torch.nn.Module
    dataset: utils.DatasetBase
    hparams: AttributeDict

    def __init__(
        self, hparams: LinearClassifierMethodParams = None, **kwargs,
    ):
        super().__init__()

        if hparams is None:
            hparams = self.params(**kwargs)
        elif isinstance(hparams, dict):
            hparams = self.params(**hparams, **kwargs)

        self.hparams = AttributeDict(attr.asdict(hparams))

        # actually do a load that is a little more flexible
        self.model = utils.get_encoder(hparams.encoder_arch)

        self.dataset = utils.get_class_dataset(hparams.dataset_name)

        self.classifier = torch.nn.Linear(hparams.embedding_dim, self.dataset.num_classes)

    def load_model_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if not k.startswith("model."):
                del state_dict[k]
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        with torch.no_grad():
            embedding = self.model(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = utils.calculate_accuracy(y_hat, y, topk=(1, 5))

        log_data = {"step_train_loss": loss, "step_train_acc1": acc1, "step_train_acc5": acc5}
        return {"loss": loss, "log": log_data}

    def validation_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        y_hat = self.forward(x)
        acc1, acc5 = utils.calculate_accuracy(y_hat, y, topk=(1, 5))
        return {
            "valid_loss": F.cross_entropy(y_hat, y),
            "valid_acc1": acc1,
            "valid_acc5": acc5,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        avg_acc1 = torch.stack([x["valid_acc1"] for x in outputs]).mean()
        avg_acc5 = torch.stack([x["valid_acc5"] for x in outputs]).mean()

        log_data = {"valid_loss": avg_loss, "valid_acc1": avg_acc1, "valid_acc5": avg_acc5}
        print(log_data)
        return {
            "val_loss": avg_loss,
            "log": log_data,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        milestones = [math.floor(self.hparams.max_epochs * 0.6), math.floor(self.hparams.max_epochs * 0.8)]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)
        return [optimizer], [self.lr_scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.dataset.get_train(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.get_validation(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
        )

    @classmethod
    def params(cls, **kwargs) -> LinearClassifierMethodParams:
        return LinearClassifierMethodParams(**kwargs)

    @classmethod
    def from_moco_checkpoint(cls, checkpoint_path, **kwargs):
        """ Loads hyperparameters and model from moco checkpoint """
        checkpoint = torch.load(checkpoint_path)
        moco_hparams = checkpoint["hyper_parameters"]
        params = cls.params(
            encoder_arch=moco_hparams["encoder_arch"],
            embedding_dim=moco_hparams["embedding_dim"],
            dataset_name=moco_hparams["dataset_name"],
            **kwargs,
        )
        model = cls(params)
        model.load_model_from_checkpoint(checkpoint_path)
        return model
