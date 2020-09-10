import os
import random
from typing import Any
from typing import Callable
from typing import Optional

import attr
import torch
import torchvision
from PIL import ImageFilter
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.datasets import ImageFolder

import ws_resnet

###################
# Transform utils #
###################


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


moco_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
moco_test_transform = transforms.Compose([transforms.ToTensor(), moco_normalize])


@attr.s(auto_attribs=True)
class MoCoTransforms:
    s: float = 0.5
    crop_size: int = 224
    apply_blur: bool = True

    def split_transform(self, img) -> torch.Tensor:
        transform = self.single_transform()
        return torch.stack((transform(img), transform(img)))

    def single_transform(self):
        transform_list = [
            transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
        ]
        if self.apply_blur:
            transform_list.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        transform_list.append(transforms.ToTensor())
        transform_list.append(moco_normalize)
        return transforms.Compose(transform_list)

    def get_test_transform(self):
        # rough heuristic for resizing test set before crop
        if self.crop_size < 128:
            test_resize = 128
        else:
            test_resize = 256
        return transforms.Compose(
            [
                transforms.Resize(test_resize),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                moco_normalize,
            ]
        )


#################
# Dataset utils #
#################


@attr.s(auto_attribs=True, slots=True)
class DatasetBase:
    _train_ds: Optional[torch.utils.data.Dataset] = None
    _validation_ds: Optional[torch.utils.data.Dataset] = None
    _test_ds: Optional[torch.utils.data.Dataset] = None
    transform_train: Optional[Callable] = None
    transform_test: Optional[Callable] = None

    def get_train(self) -> torch.utils.data.Dataset:
        if self._train_ds is None:
            self._train_ds = self.configure_train()
        return self._train_ds

    def configure_train(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def get_validation(self) -> torch.utils.data.Dataset:
        if self._validation_ds is None:
            self._validation_ds = self.configure_validation()
        return self._validation_ds

    def configure_validation(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @property
    def data_path(self):
        pathstr = os.environ.get("DATA_PATH", os.getcwd())
        os.makedirs(pathstr, exist_ok=True)
        return pathstr

    @property
    def instance_shape(self):
        img = next(iter(self.get_train()))[0]
        return img.shape

    @property
    def num_classes(self):
        train_ds = self.get_train()
        if hasattr(train_ds, "classes"):
            return len(train_ds.classes)
        return None


stl10_default_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
)


@attr.s(auto_attribs=True, slots=True)
class STL10UnlabeledDataset(DatasetBase):
    transform_train: Callable[[Any], torch.Tensor] = stl10_default_transform
    transform_test: Callable[[Any], torch.Tensor] = stl10_default_transform

    def configure_train(self):
        return STL10(self.data_path, split="train+unlabeled", download=True, transform=self.transform_train)

    def configure_validation(self):
        return STL10(self.data_path, split="test", download=True, transform=self.transform_test)


@attr.s(auto_attribs=True, slots=True)
class STL10LabeledDataset(DatasetBase):
    transform_train: Callable[[Any], torch.Tensor] = stl10_default_transform
    transform_test: Callable[[Any], torch.Tensor] = stl10_default_transform

    def configure_train(self):
        return STL10(self.data_path, split="train", download=True, transform=self.transform_train)

    def configure_validation(self):
        return STL10(self.data_path, split="test", download=True, transform=self.transform_test)


imagenet_default_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
)


@attr.s(auto_attribs=True, slots=True)
class ImagenetDataset(DatasetBase):
    transform_train: Callable[[Any], torch.Tensor] = imagenet_default_transform
    transform_test: Callable[[Any], torch.Tensor] = imagenet_default_transform

    def configure_train(self):
        assert os.path.exists(self.data_path + "/imagenet/train")
        return ImageFolder(self.data_path + "/imagenet/train", transform=self.transform_train)

    def configure_validation(self):
        assert os.path.exists(self.data_path + "/imagenet/val")
        return ImageFolder(self.data_path + "/imagenet/val", transform=self.transform_test)


def get_moco_dataset(name: str, t: MoCoTransforms) -> DatasetBase:
    if name == "stl10":
        return STL10UnlabeledDataset(transform_train=t.split_transform, transform_test=t.get_test_transform())
    elif name == "imagenet":
        return ImagenetDataset(transform_train=t.split_transform, transform_test=t.get_test_transform())
    raise NotImplementedError(f"Dataset {name} not defined")


def get_class_transforms(crop_size, resize):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    transform_test = transforms.Compose(
        [transforms.Resize(resize), transforms.CenterCrop(crop_size), transforms.ToTensor(), normalize]
    )
    return transform_train, transform_test


def get_class_dataset(name: str) -> DatasetBase:
    if name == "stl10":
        transform_train, transform_test = get_class_transforms(96, 128)
        return STL10LabeledDataset(transform_train=transform_train, transform_test=transform_test)
    elif name == "imagenet":
        transform_train, transform_test = get_class_transforms(224, 256)
        return ImagenetDataset(transform_train=transform_train, transform_test=transform_test)
    raise NotImplementedError(f"Dataset {name} not defined")


#####################
# Parallelism utils #
#####################


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class BatchShuffleDDP:
    @staticmethod
    @torch.no_grad()
    def shuffle(x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @staticmethod
    @torch.no_grad()
    def unshuffle(x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


###############
# Model utils #
###############


class MLP(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers, weight_standardization=False, normalization=None
    ):
        super().__init__()
        assert num_layers >= 0, "negative layers?!?"
        if normalization is not None:
            assert callable(normalization), "normalization must be callable"

        if num_layers == 0:
            self.net = torch.nn.Identity()
            return

        if num_layers == 1:
            self.net = torch.nn.Linear(input_dim, output_dim)
            return

        linear_net = ws_resnet.Linear if weight_standardization else torch.nn.Linear

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(linear_net(prev_dim, hidden_dim))
            if normalization is not None:
                layers.append(normalization())
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_encoder(name: str, **kwargs) -> torch.nn.Module:
    """
    Gets just the encoder portion of a torchvision model (replaces final layer with identity)
    :param name: (str) name of the model
    :param kwargs: kwargs to send to the model
    :return:
    """

    if name in ws_resnet.__dict__:
        model_creator = ws_resnet.__dict__.get(name)
    elif name in torchvision.models.__dict__:
        model_creator = torchvision.models.__dict__.get(name)
    else:
        raise AttributeError(f"Unknown architecture {name}")

    assert model_creator is not None, f"no torchvision model named {name}"
    model = model_creator(**kwargs)
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    else:
        raise NotImplementedError(f"Unknown class {model.__class__}")

    return model


####################
# Evaluation utils #
####################


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
