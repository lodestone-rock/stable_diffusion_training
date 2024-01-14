import json
from models import FlaxUNet2DConditionModel
from diffusers import UNet2DConditionModel
from transformers import FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection, CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
# import jax
import jax.numpy as jnp
from safetensors import safe_open
from safetensors.torch import save_file
import optree
import pandas as pd
import torch
import numpy as np


def dict_to_csv_pandas(dictionary, csv_file):
    df = pd.DataFrame(list(dictionary.items()), columns=['Key', 'Value'])
    df.to_csv(csv_file, index=False)

def load_from_safetensors(safetensors_file_list:list) -> dict:
    tensors = {}
    for safetensors_file in safetensors_file_list:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors

def dict_to_dot_notation(dictionary, parent_key='', sep='.'):
    """Convert a nested dictionary to dot notation."""
    items = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(dict_to_dot_notation(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def dot_notation_to_dict(dot_dict, sep='.'):
    """Convert dot notation to a nested dictionary."""
    result = {}
    for key, value in dot_dict.items():
        keys = key.split(sep)
        current_level = result
        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})
        current_level[keys[-1]] = value
    return result

def save_dict_as_json(dictionary, file_path):
    """
    Save a dictionary as a JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file)

def read_json_as_dict(file_path):
    """
    Read a JSON file and return its contents as a dictionary.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    return data

def convert_pytorch_unet_to_flax(unet_folder_path: str, output_path:str, unet_layer_mapping_path:str) -> None:
    json_file_path = f'{unet_folder_path}/config.json'

    safetensors_file_list = [
        f'{unet_folder_path}/diffusion_pytorch_model.safetensors'
    ]

    # load unet safetensors to compare it with 
    pytorch_unet_state_dict = load_from_safetensors(safetensors_file_list)
    # unet_shape_pytorch = optree.tree_map(lambda x: tuple(x.shape), pytorch_unet_state_dict)
    # unet_shape_flattened_pytorch = dict_to_dot_notation(unet_shape_pytorch)
    # dict_to_csv_pandas(unet_shape_pytorch, "pytorch_unet.csv")
    # save_dict_as_json(unet_shape_pytorch, 'unet_shape_torch.json')

    # init random unet tensor in jax
    unet = FlaxUNet2DConditionModel.from_config(read_json_as_dict(json_file_path))
    # unet_weights = unet.init_weights(jax.random.PRNGKey(42))
    # unet_weights_flattened = dict_to_dot_notation(unet_weights)
    # unet_shape = jax.tree_map(lambda x: x.shape, unet_weights)
    # unet_shape_flattened = dict_to_dot_notation(unet_shape)
    # dict_to_csv_pandas(unet_shape_flattened, "jax_unet.csv")
    # save_dict_as_json(unet_shape_flattened, 'unet_shape.json')

    layer_mapping = read_json_as_dict(unet_layer_mapping_path)

    flax_unet = {}

    for pytorch_layer, flax_layer in layer_mapping.items():

        # conv layers are transposed from HWIO into OIHW
        if pytorch_unet_state_dict[pytorch_layer].dim() == 4:
            flax_unet[flax_layer] = jnp.array(pytorch_unet_state_dict[pytorch_layer].permute(2,3,1,0).numpy(), dtype=jnp.float32)
        # GEMM layers are transposed from ND to DN
        elif pytorch_unet_state_dict[pytorch_layer].dim() == 2:
            flax_unet[flax_layer] = jnp.array(pytorch_unet_state_dict[pytorch_layer].permute(1,0).numpy(), dtype=jnp.float32)
        # bias and norm stays the same because it's just 1D tensor
        else:
            flax_unet[flax_layer] = jnp.array(pytorch_unet_state_dict[pytorch_layer].numpy(), dtype=jnp.float32)

    flax_unet = dot_notation_to_dict(flax_unet)
    unet.save_pretrained(params=flax_unet, save_directory=output_path)

def convert_pytorch_clip_to_flax(clip_folder_path: str, output_path:str, clip_layer_mapping_path:str) -> None:
    json_file_path = f'{clip_folder_path}/config.json'

    safetensors_file_list = [
        f'{clip_folder_path}/model.safetensors'
    ]

    # load clip safetensors to compare it with 
    pytorch_clip_state_dict = load_from_safetensors(safetensors_file_list)
    # clip_shape_pytorch = optree.tree_map(lambda x: tuple(x.shape), pytorch_clip_state_dict)
    # clip_shape_flattened_pytorch = dict_to_dot_notation(clip_shape_pytorch)
    # dict_to_csv_pandas(clip_shape_pytorch, "pytorch_clip.csv")
    # save_dict_as_json(clip_shape_pytorch, 'clip_shape_torch.json')

    # init random clip tensor in jax
    clip_config = CLIPTextConfig.from_json_file(json_file_path)
    clip = FlaxCLIPTextModel(config=clip_config)
    # clip_weights = clip.init_weights(jax.random.PRNGKey(42), (1,77))
    # clip_weights_flattened = dict_to_dot_notation(clip_weights)
    # clip_shape = jax.tree_map(lambda x: x.shape, clip_weights)
    # clip_shape_flattened = dict_to_dot_notation(clip_shape)
    # dict_to_csv_pandas(clip_shape_flattened, "jax_clip.csv")
    # save_dict_as_json(clip_shape_flattened, 'clip_shape.json')

    layer_mapping = read_json_as_dict(clip_layer_mapping_path)

    flax_clip = {}

    for pytorch_layer, flax_layer in layer_mapping.items():

        # embedding layer is not transposed for some reason
        if "token_embedding" in pytorch_layer or "position_embedding" in pytorch_layer:
            flax_clip[flax_layer] = jnp.array(pytorch_clip_state_dict[pytorch_layer].numpy(), dtype=jnp.float32)
        # GEMM layers are transposed from ND to DN
        elif pytorch_clip_state_dict[pytorch_layer].dim() == 2:
            flax_clip[flax_layer] = jnp.array(pytorch_clip_state_dict[pytorch_layer].permute(1,0).numpy(), dtype=jnp.float32)
        # bias and norm stays the same because it's just 1D tensor
        else:
            flax_clip[flax_layer] = jnp.array(pytorch_clip_state_dict[pytorch_layer].numpy(), dtype=jnp.float32)

    flax_clip = dot_notation_to_dict(flax_clip)
    clip.save_pretrained(params=flax_clip, save_directory=output_path)

def convert_pytorch_open_clip_to_flax(clip_folder_path: str, output_path:str, clip_layer_mapping_path:str) -> None:
    json_file_path = f'{clip_folder_path}/config.json'

    safetensors_file_list = [
        f'{clip_folder_path}/model.safetensors'
    ]

    # load clip safetensors to compare it with 
    pytorch_clip_state_dict = load_from_safetensors(safetensors_file_list)
    # clip_shape_pytorch = optree.tree_map(lambda x: tuple(x.shape), pytorch_clip_state_dict)
    # clip_shape_flattened_pytorch = dict_to_dot_notation(clip_shape_pytorch)
    # dict_to_csv_pandas(clip_shape_pytorch, "pytorch_clip.csv")
    # save_dict_as_json(clip_shape_pytorch, 'clip_shape_torch.json')

    # init random clip tensor in jax
    clip_config = CLIPTextConfig.from_json_file(json_file_path)
    clip = FlaxCLIPTextModelWithProjection(config=clip_config)
    # clip_weights = clip.init_weights(jax.random.PRNGKey(42), (1,77))
    # clip_weights_flattened = dict_to_dot_notation(clip_weights)
    # clip_shape = jax.tree_map(lambda x: x.shape, clip_weights)
    # clip_shape_flattened = dict_to_dot_notation(clip_shape)
    # dict_to_csv_pandas(clip_shape_flattened, "jax_clip.csv")
    # save_dict_as_json(clip_shape_flattened, 'clip_shape.json')

    layer_mapping = read_json_as_dict(clip_layer_mapping_path)

    flax_clip = {}

    for pytorch_layer, flax_layer in layer_mapping.items():

        # embedding layer is not transposed for some reason
        if "token_embedding" in pytorch_layer or "position_embedding" in pytorch_layer:
            flax_clip[flax_layer] = jnp.array(pytorch_clip_state_dict[pytorch_layer].numpy(), dtype=jnp.float32)
        # GEMM layers are transposed from ND to DN
        elif pytorch_clip_state_dict[pytorch_layer].dim() == 2:
            flax_clip[flax_layer] = jnp.array(pytorch_clip_state_dict[pytorch_layer].permute(1,0).numpy(), dtype=jnp.float32)
        # bias and norm stays the same because it's just 1D tensor
        else:
            flax_clip[flax_layer] = jnp.array(pytorch_clip_state_dict[pytorch_layer].numpy(), dtype=jnp.float32)

    flax_clip = dot_notation_to_dict(flax_clip)
    clip.save_pretrained(params=flax_clip, save_directory=output_path)

def convert_flax_unet_to_pytorch(model_dir: str, output_path:str, unet_layer_mapping_path:str) -> None:

    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", use_memory_efficient=False
    )

    unet_params = dict_to_dot_notation(unet_params)
    layer_mapping = read_json_as_dict(unet_layer_mapping_path)

    pytorch_unet = {}

    for pytorch_layer, flax_layer in layer_mapping.items():

        # conv layers are transposed from HWIO into OIHW
        if unet_params[flax_layer].ndim == 4:
            pytorch_unet[pytorch_layer] = torch.from_numpy(np.array(unet_params[flax_layer].transpose(3,2,0,1))).to(dtype=torch.float16)
        # GEMM layers are transposed from ND to DN
        elif unet_params[flax_layer].ndim == 2:
            pytorch_unet[pytorch_layer] = torch.from_numpy(np.array(unet_params[flax_layer].transpose(1,0))).to(dtype=torch.float16)
        # bias and norm stays the same because it's just 1D tensor
        else:
            pytorch_unet[pytorch_layer] = torch.from_numpy(np.array(unet_params[flax_layer])).to(dtype=torch.float16)
    

    unet_pytorch = UNet2DConditionModel.from_config(read_json_as_dict(f"{model_dir}/unet/config.json"))
    unet_pytorch.load_state_dict(pytorch_unet)
    unet_pytorch.save_pretrained(output_path)

def convert_flax_open_clip_to_pytorch(model_dir: str, output_path:str, clip_layer_mapping_path:str) -> None:

    text_encoder, text_encoder_params = FlaxCLIPTextModelWithProjection.from_pretrained(
        model_dir, subfolder="text_encoder_2", _do_init=False
    )

    text_encoder_params = dict_to_dot_notation(text_encoder_params)
    layer_mapping = read_json_as_dict(clip_layer_mapping_path)

    layer_mapping = read_json_as_dict(clip_layer_mapping_path)

    pytorch_clip = {}

    for pytorch_layer, flax_layer in layer_mapping.items():

        # embedding layer is not transposed for some reason
        if "token_embedding" in flax_layer or "position_embedding" in flax_layer:
            pytorch_clip[pytorch_layer] = torch.from_numpy(np.array(text_encoder_params[flax_layer])).to(dtype=torch.float16)
        # GEMM layers are transposed from ND to DN
        elif text_encoder_params[flax_layer].ndim == 2:
            pytorch_clip[pytorch_layer] = torch.from_numpy(np.array(text_encoder_params[flax_layer].transpose(1,0))).to(dtype=torch.float16)
        # bias and norm stays the same because it's just 1D tensor
        else:
            pytorch_clip[pytorch_layer] = torch.from_numpy(np.array(text_encoder_params[flax_layer])).to(dtype=torch.float16)


    clip_config = CLIPTextConfig.from_json_file(f"{model_dir}/text_encoder_2/config.json")
    clip_pytorch = CLIPTextModelWithProjection(config=clip_config)
    clip_pytorch.load_state_dict(pytorch_clip)
    clip_pytorch.save_pretrained(output_path)

def convert_flax_clip_to_pytorch(model_dir: str, output_path:str, clip_layer_mapping_path:str) -> None:

    text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
        model_dir, subfolder="text_encoder", _do_init=False
    )

    text_encoder_params = dict_to_dot_notation(text_encoder_params)
    layer_mapping = read_json_as_dict(clip_layer_mapping_path)

    layer_mapping = read_json_as_dict(clip_layer_mapping_path)

    pytorch_clip = {}

    for pytorch_layer, flax_layer in layer_mapping.items():

        # embedding layer is not transposed for some reason
        if "token_embedding" in flax_layer or "position_embedding" in flax_layer:
            pytorch_clip[pytorch_layer] = torch.from_numpy(np.array(text_encoder_params[flax_layer])).to(dtype=torch.float16)
        # GEMM layers are transposed from ND to DN
        elif text_encoder_params[flax_layer].ndim == 2:
            pytorch_clip[pytorch_layer] = torch.from_numpy(np.array(text_encoder_params[flax_layer].transpose(1,0))).to(dtype=torch.float16)
        # bias and norm stays the same because it's just 1D tensor
        else:
            pytorch_clip[pytorch_layer] = torch.from_numpy(np.array(text_encoder_params[flax_layer])).to(dtype=torch.float16)


    clip_config = CLIPTextConfig.from_json_file(f"{model_dir}/text_encoder/config.json")
    clip_pytorch = CLIPTextModel(config=clip_config)
    clip_pytorch.load_state_dict(pytorch_clip)
    clip_pytorch.save_pretrained(output_path)
