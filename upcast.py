from transformers import FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection
from models import FlaxUNet2DConditionModel
from diffusers import FlaxAutoencoderKL
import jax
import jax.numpy as jnp


model_dir = "Segmind-Vega-Flax"

text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
    model_dir, subfolder="text_encoder", dtype=jnp.float32, _do_init=False
)

text_encoder_2, text_encoder_params_2 = FlaxCLIPTextModelWithProjection.from_pretrained(
    model_dir, subfolder="text_encoder_2", dtype=jnp.float32, _do_init=False
)

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    model_dir, subfolder="unet", dtype=jnp.float32, use_memory_efficient=False
)

vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    model_dir,
    dtype=jnp.float32,
    subfolder="vae",
)

text_encoder_params = jax.tree_map(lambda x: x.astype(jnp.float32), text_encoder_params)
text_encoder_params_2 = jax.tree_map(lambda x: x.astype(jnp.float32), text_encoder_params_2)
unet_params = jax.tree_map(lambda x: x.astype(jnp.float32), unet_params)
vae_params = jax.tree_map(lambda x: x.astype(jnp.float32), vae_params)

text_encoder.save_pretrained(save_directory=f"{model_dir}/text_encoder", params=text_encoder_params)
text_encoder_2.save_pretrained(save_directory=f"{model_dir}/text_encoder_2", params=text_encoder_params_2)
unet.save_pretrained(save_directory=f"{model_dir}/unet", params=unet_params)
vae.save_pretrained(save_directory=f"{model_dir}/vae", params=vae_params)
print()