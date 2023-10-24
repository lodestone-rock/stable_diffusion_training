import json
import gc

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
import optax
from flax.training import train_state

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel
)

from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

import diffusers.schedulers.scheduling_ddim_flax

print()