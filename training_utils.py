import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from diffusers import (
    FlaxAutoencoderKL,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel
)
from schedulers import FlaxDDPMScheduler
from transformers import  CLIPTokenizer, FlaxCLIPTextModel
from flax.training import train_state
import optax
from flax import struct
from typing import Callable


class FrozenModel(struct.PyTreeNode):
    """
    mimic the behaviour of train_state but this time for frozen params 
    to make it passable to the jitted function
    """
    
    # use pytree_node=False to indicate an attribute should not be touched
    # by Jax transformations.
    call: Callable = struct.field(pytree_node=False)
    params: dict = struct.field(pytree_node=True)

    @classmethod
    def create(cls,apply_fn, params):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        return cls(
            call=call,
            params=params,
        )


@dataclass
class TrainingConfig:
    """
    reading model properties from json. i should modify the json to when the model is done training
    format:
    {
        "model_path":"model_checkpoints/path"
        "learning_rate": 1e-6,
        "lr_scheduler": "constant",
        "adam_to_lion_scale_factor": 7.0
    }
    """
    model_path: str
    learning_rate: float
    unet_learning_rate: float
    text_encoder_learning_rate: float
    lr_scheduler: str
    adam_to_lion_scale_factor: float





def calculate_resolution_array(max_res_area=512 ** 2, bucket_lower_bound_res=256, rounding=64):
    """
    helper function to calculate image bucket 

    Parameters:
    - max_res_area (int): The maximum target resolution area of the image.
    - bucket_lower_bound_res (int): minimum minor axis (smaller axis).
    - rounding (int): rounding steps / rounding increment.

    Returns:
    - resolution (numpy.ndarray): A 2D NumPy array representing the resolution pairs (width, height).
    """
    root_max_res = max_res_area ** (1 / 2)
    centroid = int(root_max_res)

    # a sequence of number that divisible by 64 with constraint
    w = np.arange(bucket_lower_bound_res // rounding * rounding, centroid // rounding * rounding + rounding, rounding)
    # y=1/x formula with rounding down to the nearest multiple of 64
    # will maximize the clamped resolution to maximum res area
    h = ((max_res_area / w) // rounding * rounding).astype(int)

    # is square array possible? if so chop the last bit before combining
    if w[-1] - h[-1] == 0:
        w_delta = np.flip(w[:-1])
        h_delta = np.flip(h[:-1])

    w = np.concatenate([w,w_delta])
    h = np.concatenate([h,h_delta])

    resolution = np.stack([w,h]).T

    return resolution


def load_models(model_dir:str) -> dict:
    """
    Load models from a directory using HuggingFace. the config hard coded for now!

    Args:
        model_dir (str): The path to the directory containing the models.

    Returns:
        dict: A dictionary containing the loaded models and their parameters.
            {
                "unet":{
                    "unet_params": unet_params,
                    "unet_model": unet,
                },
                "vae":{
                    "vae_params": vae_params,
                    "vae_model": vae,
                },
                "text_encoder":{
                    "text_encoder_params": text_encoder_params,
                    "text_encoder_model": text_encoder,
                },
                "schedulers":{
                    "noise_scheduler_state": noise_scheduler_state,
                    "noise_scheduler_object": noise_scheduler,
                },
            }
    """

    # load the model params and model object

    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", dtype=jnp.bfloat16, use_memory_efficient=True
    )
    text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
        model_dir, subfolder="text_encoder", dtype=jnp.bfloat16, _do_init=False
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        model_dir,
        dtype=jnp.bfloat16,
        subfolder="vae",
    )
    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="zero_snr_scaled_linear",
        num_train_timesteps=1000,
        prediction_type="v_prediction",
    )
    noise_scheduler_state = noise_scheduler.create_state()

    #should've put this in dataclasses 
    return{
        "unet":{
            "unet_params": unet_params,
            "unet_model": unet,
        },
        "vae":{
            "vae_params": vae_params,
            "vae_model": vae,
        },
        "text_encoder":{
            "text_encoder_params": text_encoder_params,
            "text_encoder_model": text_encoder,
        },
        "schedulers":{
            "noise_scheduler_state": noise_scheduler_state,
            "noise_scheduler_object": noise_scheduler,
        },
    }


def create_frozen_states(models: dict):
    """
    create frozen training states that bundled with teh function or method associated with it

    Args:
        models (dict): A dictionary containing models and parameters.

    Returns:
        dict: A dictionary containing the optimizer states for U-Net and text encoder models.
            {
                "vae_state": vae_state,
                "schedulers_state": schedulers_state
            }
    """
    
    vae_state = FrozenModel(
        call=models["vae"]["vae_model"],
        params=models["vae"]["vae_params"],
    )

    schedulers_state = FrozenModel(
        # welp not a function but eh it should works
        call=models["schedulers"]["noise_scheduler_object"],
        params=models["schedulers"]["noise_scheduler_state"],
    )
    return {
        
        "vae_state": vae_state,
        "schedulers_state": schedulers_state
    }


def create_lion_optimizer_states(
    models: dict,
    train_unet: bool = True,
    train_text_encoder: bool = True,
    adam_to_lion_scale_factor: int = 7,
    u_net_learning_rate: float = 1e-6,
    text_encoder_learning_rate: float = 1e-6,
):
    """
    Create optimizer states for Lion, a custom optimizer, for U-Net and CLIP text encoder models.

    Args:
        models (dict): A dictionary containing the U-Net and text encoder models and parameters.
            {
                "unet": {
                    "unet_model": your_unet_model,
                    "unet_params": your_unet_params,
                },
                "text_encoder": {
                    "text_encoder_model": your_text_encoder_model,
                    "text_encoder_params": your_text_encoder_params,
                }
            }
        train_unet (bool): Whether to train the U-Net model.
        train_text_encoder (bool): Whether to train the text encoder model.
        adam_to_lion_scale_factor (int): Scaling factor for adjusting learning rates.
        u_net_learning_rate (float): unet learning rate 
        text_encoder_learning_rate (float): text encoder learning rate

    Returns:
        dict: A dictionary containing the optimizer states for U-Net and text encoder models.
            {
                "unet_state": unet_state or None,
                "text_encoder_state": text_encoder_state or None
            }


    """
    # no fancy optimizer atm just use linear constant lr
    # optimizer for U-Net
    # use this context manager to ensure all of this ops happening in CPU
    # so it does not waste precious HBM space in TPU

    unet_state = None
    text_encoder_state = None

    with jax.default_device(jax.devices("cpu")[0]):

        if train_unet:
            u_net_constant_scheduler = optax.constant_schedule(
                u_net_learning_rate / adam_to_lion_scale_factor
            )
            u_net_lion = optax.lion(
                learning_rate=u_net_constant_scheduler,
                b1=0.9,
                b2=0.99,
                weight_decay=1e-2 * adam_to_lion_scale_factor,
            )
            u_net_optimizer = optax.chain(
                optax.clip_by_global_norm(1),  # prevent explosion
                u_net_lion,
            )
            unet_state = train_state.TrainState.create(
                apply_fn=models["unet"]["unet_model"].apply,
                params=models["unet"]["unet_params"],
                tx=u_net_optimizer,
            )

        # optimizer for CLIP text encoder
        if train_text_encoder:
            text_encoder_constant_scheduler = optax.constant_schedule(
                text_encoder_learning_rate / adam_to_lion_scale_factor
            )
            text_encoder_lion = optax.lion(
                learning_rate=text_encoder_constant_scheduler,
                b1=0.9,
                b2=0.99,
                weight_decay=1e-2 * adam_to_lion_scale_factor,
            )
            text_encoder_optimizer = optax.chain(
                optax.clip_by_global_norm(1),  # prevent explosion
                text_encoder_lion,
            )
            text_encoder_state = train_state.TrainState.create(
                # transformer implementation does not have apply method apparently
                apply_fn=models["text_encoder"]["text_encoder_model"].__call__,
                params=models["text_encoder"]["text_encoder_params"],
                tx=text_encoder_optimizer,
            )
    
    return {
        "unet_state": unet_state,
        "text_encoder_state": text_encoder_state
    }