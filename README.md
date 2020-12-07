# Pytorch-Lightning Implementation of Self-Supervised algorithms

This is an implementation of [MoCo](https://arxiv.org/abs/1911.05722), [MoCo v2](https://arxiv.org/abs/2003.04297), and [BYOL](https://arxiv.org/abs/2006.07733) using [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). The configuration can be tweaked to implement a range of possible self-supervised implementations.

We have recently added the necessary functionality to run [SimCLR](https://arxiv.org/abs/2002.05709) and [EqCo](https://arxiv.org/abs/2010.01929) for comparison.

See the blog post [Understanding self-supervised and contrastive learning with "Bootstrap Your Own Latent" (BYOL)](https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html) for more details.

## Install

Make sure you're in a fresh `conda` or `venv` environment, then run:

```bash
git clone https://github.com/untitled-ai/self_supervised
cd self_supervised
pip install -r requirements.txt
```

## Replicating our results

You can replicate the results of our blog post by running `python train_blog.py`. The cosine similarity between z and z' is reported as `step_neg_cos` (for negative examples) and `step_pos_cos` (for positive examples). Classification accuracy is reported as `valid_class_acc`.

## Getting started

To get started with training a ResNet-18 with MoCo v2 on STL-10 (the default configuration):

```python
import os
import pytorch_lightning as pl
from moco import MoCoMethod 

os.environ["DATA_PATH"] = "~/data"

model = MoCoMethod()
trainer = pl.Trainer(gpus=1, max_epochs=320)    
trainer.fit(model) 
trainer.save_checkpoint("example.ckpt")
```

During training, the top1/top5 accuracies (out of 1+K examples) are reported where possible.
 
During validation, an `sklearn` linear classifier is trained on half the test set and validated on the other half. The top1 accuracy is logged as `train_class_acc` / `valid_class_acc`. 

To train a linear classifier on the result:

```python
import pytorch_lightning as pl
from linear_classifier import LinearClassifierMethod
linear_model = LinearClassifierMethod.from_moco_checkpoint("example.ckpt")
trainer = pl.Trainer(gpus=1, max_epochs=100)    

trainer.fit(linear_model)
```

## BYOL

To train BYOL rather than MoCo v2, use the following parameters:

```python
from moco import MoCoMethodParams
from moco import MoCoMethod
params = MoCoMethodParams(
    prediction_mlp_layers = 2,
    mlp_normalization = "bn",
    loss_type = "ip",
    use_negative_examples_from_queue = False,
    use_both_augmentations_as_queries = True,
    use_momentum_schedule = True,
    optimizer_name = "lars",
    exclude_matching_parameters_from_lars = [".bias", ".bn"],
    loss_constant_factor = 2
)
model = MoCoMethod(params)
```

For convenience, you can instead pass these parameters add keyword args, for example with `model = MoCoMethod(batch_size=128)`.

## SimCLR

To run SimCLR, simply `use_negative_examples_from_batch` and disable `use_negative_examples_from_queue`. You can also set `K=0`: 

 ```python
hparams = MoCoMethodParams(
    use_negative_examples_from_batch=True,
    use_negative_examples_from_queue=False,
    K=0,
    m=0.0,
    use_both_augmentations_as_queries=True,
)
```

This is not super efficient as it still has the separate offline network, and so it does each embedding calculation twice. However, it is sufficient for comparing the methods with each other.

**Note for multi-GPU setups**: this currently only uses negatives on the same GPU, and will not sync negatives across multiple GPUs. 

## EqCo

Using EqCo increases the weight on negative examples to create an "effective" number of negative examples `eqco_alpha`. To train MoCo with a smaller queue but the same effective number, you could use,

```python
hparams = MoCoMethodParams(use_eqco_margin=True, eqco_alpha=65536, K=256)
```

We found this to be effective in matching the performance of the larger queue in our testing.  Because we are only using 256 examples now, we can actually just get them from the batch,

```python
hparams = MoCoMethodParams(
    use_eqco_margin=True, 
    eqco_alpha=65536, 
    K=0,
    use_negative_examples_from_batch=True,
    use_negative_examples_from_queue=False 
)
```

We've found taking negatives from the same batch to perform marginally better than MoCo on STL10.

## Training results

Training a ResNet-18 for 320 epochs on STL-10 achieved 85% linear classification accuracy on the test set (1 fold of 5000). This used all default parameters.

 Training a ResNet-50 for 200 epochs on ImageNet achieves 65.6% linear classification accuracy on the test set. 
 This used 8 gpus with `ddp` and parameters:
 
 ```python
hparams = MoCoMethodParams(
    encoder_arch="resnet50",
    shuffle_batch_norm=True,
    embedding_dim=2048,
    mlp_hidden_dim=2048,
    dataset_name="imagenet",
    batch_size=32,
    lr=0.03,
    max_epochs=200,
    transform_crop_size=224,
    num_data_workers=32,
    gather_keys_for_queue=True,
)
```

(the `batch_size` differs from the moco documentation due to the way Pytorch-Lightning handles multi-gpu 
training in `ddp` -- the effective number is `batch_size=256`). **Note that for ImageNet we suggest using 
`val_percent_check=0.1` when calling `pl.Trainer`** to reduce the time fitting the sklearn model.
 

## Other options

All possible `hparams` for MoCoMethod, along with defaults:

```python
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
    m: float = 0.996
    T: float = 0.2
    gather_keys_for_queue: bool = False

    # eqco parameters
    eqco_alpha: int = 65536
    use_eqco_margin: bool = False
    use_negative_examples_from_batch: bool = False

    # optimization parameters
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_epochs: int = 200

    # transform parameters
    transform_s: float = 0.5
    transform_crop_size: int = 96
    transform_apply_blur: bool = True

    # Change these to make more like BYOL
    use_momentum_schedule: bool = False
    loss_type: str = "ce"
    use_negative_examples_from_queue: bool = True
    use_both_augmentations_as_queries: bool = False
    optimizer_name: str = "sgd"
    exclude_matching_parameters_from_lars: List[str] = []  # set to [".bias", ".bn"] to match paper
    loss_constant_factor: float = 1

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
```

A few options require more explanation:

**encoder_arch** can be any torchvision model, or can be one of the ResNet models with weight standardization defined in 
`ws_resnet.py`.

**dataset_name** can be `stl10` or `imagenet`. `os.environ["DATA_PATH"]` will be used as the path to the data. STL-10 will
be downloaded if it does not already exist.

**loss_type** can be `ce` (cross entropy) with one of the `use_negative_examples` to correspond to MoCo or `ip` (inner product) 
with both `use_negative_examples=False` to correspond to BYOL. It can also be `bce`, which is similar to `ip` but applies the 
binary cross entropy loss function to the result.

**optimizer_name**, currently just `sgd` or `lars`. 

**exclude_matching_parameters_from_lars** will remove weight decay and LARS learning rate from matching parameters. Set
to `[".bias", ".bn"]` to match BYOL paper implementation.

**mlp_normalization** can be None for no normalization, `bn` for batch normalization, `ln` for layer norm, `gn` for group
norm, or `br` for [batch renormalization](https://github.com/ludvb/batchrenorm).

**prediction_mlp_normalization** defaults to `same` to use the same normalization as above, but can be given any of the
above parameters to use a different normalization.

**shuffle_batch_norm** and **gather_keys_for_queue** are both related to multi-gpu training. **shuffle_batch_norm** 
will shuffle the *key* images among GPUs, which is needed for training if batch norm is used. **gather_keys_for_queue** 
will gather key projections (z' in the blog post) from all gpus to add to the MoCo queue.
