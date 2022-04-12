from functools import partial
from typing import List
from typing import Optional

import attr


@attr.s(auto_attribs=True)
class ModelParams:
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
    m: float = 0.996
    T: float = 0.2

    # eqco parameters
    eqco_alpha: int = 65536
    use_eqco_margin: bool = False
    use_negative_examples_from_batch: bool = False

    # optimization parameters
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_epochs: int = 320
    final_lr_schedule_value: float = 0.0

    # transform parameters
    transform_s: float = 0.5
    transform_apply_blur: bool = True

    # Change these to make more like BYOL
    use_momentum_schedule: bool = False
    loss_type: str = "ce"
    use_negative_examples_from_queue: bool = True
    use_both_augmentations_as_queries: bool = False
    optimizer_name: str = "sgd"
    lars_warmup_epochs: int = 1
    lars_eta: float = 1e-3
    exclude_matching_parameters_from_lars: List[str] = []  # set to [".bias", ".bn"] to match paper
    loss_constant_factor: float = 1

    # Change these to make more like VICReg
    use_vicreg_loss: bool = False
    use_lagging_model: bool = True
    use_unit_sphere_projection: bool = True
    invariance_loss_weight: float = 25.0
    variance_loss_weight: float = 25.0
    covariance_loss_weight: float = 1.0
    variance_loss_epsilon: float = 1e-04

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


# Differences between these parameters and those used in the paper (on image net):
# max_epochs=1000,
# lr=1.6,
# batch_size=2048,
# weight_decay=1e-6,
# mlp_hidden_dim=8192,
# dim=8192,
VICRegParams = partial(
    ModelParams,
    use_vicreg_loss=True,
    loss_type="vic",
    use_lagging_model=False,
    use_unit_sphere_projection=False,
    use_negative_examples_from_queue=False,
    optimizer_name="lars",
    exclude_matching_parameters_from_lars=[".bias", ".bn"],
    projection_mlp_layers=3,
    final_lr_schedule_value=0.002,
    mlp_normalization="bn",
    lars_warmup_epochs=10,
)

BYOLParams = partial(
    ModelParams,
    prediction_mlp_layers=2,
    mlp_normalization="bn",
    loss_type="ip",
    use_negative_examples_from_queue=False,
    use_both_augmentations_as_queries=True,
    use_momentum_schedule=True,
    optimizer_name="lars",
    exclude_matching_parameters_from_lars=[".bias", ".bn"],
    loss_constant_factor=2,
)

SimCLRParams = partial(
    ModelParams,
    use_negative_examples_from_batch=True,
    use_negative_examples_from_queue=False,
    K=0,
    m=0.0,
    use_both_augmentations_as_queries=True,
)
