import copy
import math
import warnings
from functools import partial
from typing import List
from typing import Optional
from typing import Union

import attr
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

import utils
from batchrenorm import BatchRenorm1d
from lars import LARS


@attr.s(auto_attribs=True)
class MoCoMethodParams:
    # encoder model selection
    encoder_arch: str = "resnet18"
    shuffle_batch_norm: bool = False
    embedding_dim: int = 512  # must match embedding dim of encoder

    # data-related parameters
    dataset_name: str = "stl10"
    batch_size: int = 256

    # MoCo parameters
    K: int = 65536  # number of examples in queue
    dim: int = 128
    m: float = 0.999
    T: float = 0.2

    # optimization parameters
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_epochs: int = 320

    # transform parameters
    transform_s: float = 0.5
    transform_crop_size: int = 96
    transform_apply_blur: bool = True

    # Change these to make more like BYOL
    use_momentum_schedule: bool = False
    loss_type: str = "ce"
    use_negative_examples: bool = True
    use_both_augmentations_as_queries: bool = False
    optimizer_name: str = "sgd"
    exclude_matching_parameters_from_lars: List[str] = []  # set to [".bias", ".bn"] to match paper

    # MLP parameters
    projection_mlp_layers: int = 2
    prediction_mlp_layers: int = 0
    mlp_hidden_dim: int = 512

    mlp_normalization: Optional[str] = None
    prediction_mlp_normalization: Optional[str] = "same"  # if same will use mlp_normalization
    use_mlp_weight_standardization: bool = False

    # data loader parameters
    num_data_workers: int = 4
    drop_last_batch: bool = True
    pin_data_memory: bool = True
    gather_keys_for_queue: bool = False


def get_mlp_normalization(hparams: MoCoMethodParams, prediction=False):
    normalization_str = hparams.mlp_normalization
    if prediction and hparams.prediction_mlp_normalization is not "same":
        normalization_str = hparams.prediction_mlp_normalization

    if normalization_str is None:
        return None
    elif normalization_str == "bn":
        return partial(torch.nn.BatchNorm1d, num_features=hparams.mlp_hidden_dim)
    elif normalization_str == "br":
        return partial(BatchRenorm1d, num_features=hparams.mlp_hidden_dim)
    elif normalization_str == "ln":
        return partial(torch.nn.LayerNorm, normalized_shape=[hparams.mlp_hidden_dim])
    elif normalization_str == "gn":
        return partial(torch.nn.GroupNorm, num_channels=hparams.mlp_hidden_dim, num_groups=32)
    else:
        raise NotImplementedError(f"mlp normalization {normalization_str} not implemented")


class MoCoMethod(pl.LightningModule):
    model: torch.nn.Module
    dataset: utils.DatasetBase
    hparams: AttributeDict
    embedding_dim: Optional[int]

    def __init__(
        self, hparams: Union[MoCoMethodParams, dict, None] = None, **kwargs,
    ):
        super().__init__()

        if hparams is None:
            hparams = self.params(**kwargs)
        elif isinstance(hparams, dict):
            hparams = self.params(**hparams, **kwargs)

        self.hparams = AttributeDict(attr.asdict(hparams))

        # Check for configuration issues
        if (
            hparams.gather_keys_for_queue
            and not hparams.shuffle_batch_norm
            and not hparams.encoder_arch.startswith("ws_")
        ):
            warnings.warn(
                "Configuration suspicious: gather_keys_for_queue without shuffle_batch_norm or weight standardization"
            )

        if hparams.loss_type == "ce" and not hparams.use_negative_examples:
            warnings.warn("Configuration suspicious: cross entropy loss without negative examples")

        # Create encoder model
        self.model = utils.get_encoder(hparams.encoder_arch)

        # Create dataset
        transforms = utils.MoCoTransforms(
            s=hparams.transform_s, crop_size=hparams.transform_crop_size, apply_blur=hparams.transform_apply_blur
        )
        self.dataset = utils.get_moco_dataset(hparams.dataset_name, transforms)

        # "key" function (no grad)
        self.lagging_model = copy.deepcopy(self.model)
        for param in self.lagging_model.parameters():
            param.requires_grad = False

        self.projection_model = utils.MLP(
            hparams.embedding_dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.projection_mlp_layers,
            normalization=get_mlp_normalization(hparams),
            weight_standardization=hparams.use_mlp_weight_standardization,
        )

        self.prediction_model = utils.MLP(
            hparams.dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.prediction_mlp_layers,
            normalization=get_mlp_normalization(hparams, prediction=True),
            weight_standardization=hparams.use_mlp_weight_standardization,
        )

        #  "key" function (no grad)
        self.lagging_projection_model = copy.deepcopy(self.projection_model)
        for param in self.lagging_projection_model.parameters():
            param.requires_grad = False

        # this classifier is used to compute representation quality each epoch
        self.sklearn_classifier = LogisticRegression(max_iter=100, solver="liblinear")

        # create the queue
        self.register_buffer("queue", torch.randn(hparams.dim, hparams.K))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _get_embeddings(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        bsz, nd, nc, nh, nw = x.shape
        assert nd == 2, "second dimension should be the split image -- dims should be N2CHW"
        im_q = x[:, 0].contiguous()
        im_k = x[:, 1].contiguous()

        # compute query features
        emb_q = self.model(im_q)
        q_projection = self.projection_model(emb_q)
        q = self.prediction_model(q_projection)  # queries: NxC
        q = torch.nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if self.hparams.shuffle_batch_norm:
                im_k, idx_unshuffle = utils.BatchShuffleDDP.shuffle(im_k)

            k = self.lagging_projection_model(self.lagging_model(im_k))  # keys: NxC
            k = torch.nn.functional.normalize(k, dim=1)

            if self.hparams.shuffle_batch_norm:
                k = utils.BatchShuffleDDP.unshuffle(k, idx_unshuffle)

        return emb_q, q, k

    def _get_contrastive_predictions(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        if self.hparams.use_negative_examples:
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        return logits, labels

    def _get_pos_neg_ip(self, emb_q, k):
        with torch.no_grad():
            z = self.projection_model(emb_q)
            z = torch.nn.functional.normalize(z, dim=1)
            ip = torch.mm(z, k.T)
            eye = torch.eye(z.shape[0]).to(z.device)
            pos_ip = (ip * eye).sum() / z.shape[0]
            neg_ip = (ip * (1 - eye)).sum() / (z.shape[0] * (z.shape[0] - 1))

        return pos_ip, neg_ip

    def _get_contrastive_loss(self, logits, labels):
        if self.hparams.loss_type == "ce":
            return F.cross_entropy(logits / self.hparams.T, labels)

        new_labels = torch.zeros_like(logits)
        new_labels.scatter_(1, labels.unsqueeze(1), 1)
        if self.hparams.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits / self.hparams.T, new_labels) * logits.shape[1]

        if self.hparams.loss_type == "ip":
            # inner product
            # negative sign for label=1 (maximize ip), positive sign for label=0 (minimize ip)
            inner_product = (1 - new_labels * 2) * logits
            return torch.mean((inner_product + 1).sum(dim=-1))

        raise NotImplementedError(f"Loss function {self.hparams.loss_type} not implemented")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, class_labels = batch  # batch is a tuple, we just want the image

        emb_q, q, k = self._get_embeddings(x)
        logits, labels = self._get_contrastive_predictions(q, k)
        pos_ip, neg_ip = self._get_pos_neg_ip(emb_q, k)

        contrastive_loss = self._get_contrastive_loss(logits, labels)

        if self.hparams.use_both_augmentations_as_queries:
            x_flip = torch.flip(x, dims=[1])
            emb_q2, q2, k2 = self._get_embeddings(x_flip)
            logits2, labels2 = self._get_contrastive_predictions(q2, k2)

            pos_ip2, neg_ip2 = self._get_pos_neg_ip(emb_q2, k2)
            pos_ip = (pos_ip + pos_ip2) / 2
            neg_ip = (neg_ip + neg_ip2) / 2
            contrastive_loss += self._get_contrastive_loss(logits2, labels2)

        contrastive_loss = contrastive_loss.mean()

        log_data = {"step_train_loss": contrastive_loss, "step_pos_cos": pos_ip, "step_neg_cos": neg_ip}

        with torch.no_grad():
            self._momentum_update_key_encoder()

        if self.hparams.use_negative_examples:
            acc1, acc5 = utils.calculate_accuracy(logits, labels, topk=(1, 5))
            log_data.update({"step_train_acc1": acc1, "step_train_acc5": acc5})

        # dequeue and enqueue
        if self.hparams.use_negative_examples:
            self._dequeue_and_enqueue(k)

        return {"loss": contrastive_loss, "log": log_data}

    def validation_step(self, batch, batch_idx):
        x, class_labels = batch
        with torch.no_grad():
            emb = self.model(x)

        return {"emb": emb, "labels": class_labels}

    def validation_epoch_end(self, outputs):
        embeddings = torch.cat([x["emb"] for x in outputs]).cpu().detach().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().detach().numpy()
        num_split_linear = embeddings.shape[0] // 2
        self.sklearn_classifier.fit(embeddings[:num_split_linear], labels[:num_split_linear])
        train_accuracy = self.sklearn_classifier.score(embeddings[:num_split_linear], labels[:num_split_linear]) * 100
        valid_accuracy = self.sklearn_classifier.score(embeddings[num_split_linear:], labels[num_split_linear:]) * 100

        log_data = {
            "epoch": self.current_epoch,
            "train_class_acc": train_accuracy,
            "valid_class_acc": valid_accuracy,
            "T": self._get_temp(),
            "m": self._get_m(),
        }
        print(f"Epoch {self.current_epoch} accuracy: train: {train_accuracy:.1f}%, validation: {valid_accuracy:.1f}%")
        print(log_data)
        return {
            "log": log_data,
            "train_class_acc": train_accuracy,
            "valid_class_acc": valid_accuracy,
        }

    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        regular_parameters = []
        regular_parameter_names = []
        excluded_parameters = []
        excluded_parameter_names = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad is False:
                continue
            if any(x in name for x in self.hparams.exclude_matching_parameters_from_lars):
                excluded_parameters.append(parameter)
                excluded_parameter_names.append(name)
            else:
                regular_parameters.append(parameter)
                regular_parameter_names.append(name)

        param_groups = [
            {"params": regular_parameters, "names": regular_parameter_names, "use_lars": True},
            {"params": excluded_parameters, "names": excluded_parameter_names, "use_lars": False, "weight_decay": 0,},
        ]
        if self.hparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer_name == "lars":
            optimizer = LARS
        else:
            raise NotImplementedError(f"No such optimizer {self.hparams.optimizer_name}")

        encoding_optimizer = optimizer(
            param_groups, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoding_optimizer, self.hparams.max_epochs, eta_min=0
        )
        return [encoding_optimizer], [self.lr_scheduler]

    def _get_m(self):
        if self.hparams.use_momentum_schedule is False:
            return self.hparams.m
        return 1 - (1 - self.hparams.m) * (math.cos(math.pi * self.current_epoch / self.hparams.max_epochs) + 1) / 2

    def _get_temp(self):
        return self.hparams.T

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        m = self._get_m()
        for param_q, param_k in zip(self.model.parameters(), self.lagging_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        for param_q, param_k in zip(self.projection_model.parameters(), self.lagging_projection_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.hparams.gather_keys_for_queue:
            keys = utils.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.K  # move pointer

        self.queue_ptr[0] = ptr

    def prepare_data(self) -> None:
        self.dataset.get_train()
        self.dataset.get_validation()

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
    def params(cls, **kwargs) -> MoCoMethodParams:
        return MoCoMethodParams(**kwargs)
