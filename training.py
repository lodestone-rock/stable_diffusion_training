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
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel
)
from streamer.utils import read_json_file
from schedulers import FlaxDDPMScheduler
from transformers import  CLIPTokenizer, FlaxCLIPTextModel

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    reading model properties from json. i should modify the json to when the model is done training
    format:
    {
        "model_path":"model_checkpoints/path"
        "learning_rate": 1e-6,
        "lr_scheduler": "constant"
    }
    """
    model_path: str
    learning_rate: float
    lr_scheduler: str



def load_models(model_dir:str) -> dict:
    """
    loads model from directory using HuggingFace for now
    all config are hard coded atm need to be extracted!
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





training_config = TrainingConfig(
    **read_json_file(
        "model_properties.json"
    )
)


model_path = training_config.model_path


from transformers import CLIPTokenizer
from streamer.dataloader import DataLoader
import time
import gc

from streamer.utils import (
    numpy_to_pil_and_save,
    write_list_to_file,
    TimingContextManager,
)

# debug test stuff
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

dataloader = DataLoader(
    tokenizer_obj=tokenizer,
    config="repo.json",  # Replace Any with the actual type of creds_data
    ramdisk_path="ramdisk",
    chunk_number=0,  # This should be incremented for each successful data loading
    seed=42,  # This should be incremented when all batches are processed
    training_batch_size=8,
    repeat_batch=5,
    maximum_resolution_areas=[
        576**2,
        704**2,
        832**2,
        960**2,
        1088**2,
    ],
    bucket_lower_bound_resolutions=[384, 512, 576, 704, 832],
    numb_of_worker_thread=20,
    queue_get_timeout=10,
)

dataloader._print_debug = False
dataloader.delete_prev_chunks(prev_chunk=1)
dataloader.grab_and_prefetch_chunk(
    numb_of_prefetched_batch=1,
    chunk_number=2,
)  # TODO: chunk number should be defined here so the thread is not terminated i think?
dataloader.prepare_training_dataframe()
dataloader.create_training_dataframe()
dataloader._bulk_batch_count = 400  # debug limit to 100 batch
dataloader.dispatch_worker()
with TimingContextManager("total queue"):
    for count in range(int(dataloader.total_batch)):
        with TimingContextManager(f"queue latency at batch {count}"):
            test = dataloader.grab_next_batch()
            if test == "end_of_batch":
                break
            # try:
                # text = []
                # for x, token in enumerate(test["input_ids"]):
                #     text.append(
                #         str(x)
                #         + " === "
                #         + tokenizer.decode(token.reshape(-1))
                #         .replace("<|endoftext|>", "")
                #         .replace("<|startoftext|>", "")
                #     )
                # write_list_to_file(text, f"{count}.txt")

                # for x, np_image in enumerate(test["pixel_values"]):
                #     numpy_to_pil_and_save(np_image, f"{count}-{x}-pil.png")

            # print(count, "shape", test["pixel_values"].shape)
            # # print("shape", test["input_ids"].shape)
            # except:
            #     print(f"batch {count} is none")
            if count % int(dataloader.total_batch):
                gc.collect()
            time.sleep(0.01)
print()


print()