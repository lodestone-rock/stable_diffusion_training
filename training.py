import time
import jax
import jax.numpy as jnp
import numpy as np
from training_utils import (
    TrainingConfig,
    on_device_model_training_state,
    dp_compile_all_unique_resolution,
)
from streamer.dataloader import DataLoader
from streamer.utils import (
    numpy_to_pil_and_save,
    write_list_to_file,
    TimingContextManager,
    read_json_file,
    save_dict_to_json,
)
from transformers import CLIPTokenizer
from dataclasses import fields
from tqdm.auto import tqdm

# this configuration dict iss all you need to setup this training script
config_dict_path = "model_properties.json"
config_dict = read_json_file(config_dict_path)
save_dict_to_json(config_dict, f"backup_{config_dict_path}")
# create backup json because this json is gonna used to store states

# some assertion check to check if the config is correct
assert len(config_dict["image_area_root"]) == len(
    config_dict["minimum_axis_length"]
), "number of elements in image_area_root and minimum_axis_length is not match! check your config files!"


# put all config that's neccessary for model loading here
# the rest stays in config_dict for now
training_config = TrainingConfig(
    **{key.name: config_dict[key.name] for key in fields(TrainingConfig)}
)


### dataloader init ###


# this dataloader does not use dataclass so just grab the value as is from the json
tokenizer = CLIPTokenizer.from_pretrained(
    config_dict["model_path"], subfolder="tokenizer"
)
dataloader = DataLoader(
    tokenizer_obj=tokenizer,
    config=config_dict_path,  # Replace Any with the actual type of creds_data
    ramdisk_path=config_dict[
        "ramdisk_path"
    ],  # temp storage path preferably ramdisk because it's fastlr_scheduler
    training_batch_size=config_dict["batch_size"],  # trainig batch
    repeat_batch=config_dict[
        "repeat_batch"
    ],  # number of same resolution repeated during data shuffling (prevent jax switching compiled function back and forth)
    maximum_resolution_areas=[
        x**2 for x in config_dict["image_area_root"]
    ],  # bucketing area target
    bucket_lower_bound_resolutions=config_dict[
        "minimum_axis_length"
    ],  # bucketing minimum resolution axis list
    numb_of_worker_thread=config_dict[
        "numb_of_dataloader_worker_thread"
    ],  # dont set this too high it will consume a lot of RAM
    queue_get_timeout=config_dict[
        "queue_get_timeout"
    ],  # timeout to indicate end of batch
    # init value and will get incremented inside loop
    chunk_number=config_dict[
        "chunk_number"
    ],  # This should be incremented for each successful data loading
    seed=config_dict[
        "master_seed"
    ],  # This should be incremented when all batches are processed
)

# disable debug print
if not config_dict["DEBUG"]:
    dataloader._print_debug = False

train_rngs = jax.random.PRNGKey(config_dict["master_seed"])
# create model states to device
# this should run once and let the giant loop update this variable continuously
(
    unet_state,
    text_encoder_state,
    frozen_vae,
    frozen_schedulers,
) = on_device_model_training_state(training_config)

# this should run inside giant loop

for _ in range(config_dict["chunk_limit"]):
    # delete prev chunk if any to prevent filling up the temp storage
    dataloader.delete_prev_chunks(prev_chunk=config_dict["chunk_number"] - 1)
    # reset chunk counter if it hit the limit or overflowing
    if config_dict["chunk_number"] >= config_dict["chunk_limit"]:
        dataloader.delete_prev_chunks(prev_chunk=config_dict["chunk_number"])
        config_dict["chunk_number"] = 0
    chunk = config_dict[
        "chunk_number"
    ]  # this state is stored in dict and will get updated
    dataloader.chunk_number = chunk
    # grab current chunk and next chunk concurently (or possibly more)
    dataloader.grab_and_prefetch_chunk(
        numb_of_prefetched_batch=config_dict["numb_of_prefetched_batch"],
    )  # TODO: chunk number should be defined here so the thread is not terminated i think?
    dataloader.prepare_training_dataframe()
    dataloader.create_training_dataframe()
    if config_dict["DEBUG"]:
        dataloader._bulk_batch_count = 100  # debug limit to 100 batch
    dataloader.dispatch_worker()

    # compile all posible resolution bucket
    train_step_funcs = dp_compile_all_unique_resolution(
        unet_state, text_encoder_state, frozen_vae, frozen_schedulers, training_config
    )

    # progress bar
    train_step_progress_bar = tqdm(
        total=int(dataloader._bulk_batch_count + dataloader._first_batch_count),
        desc="Training...",
        position=1,
        smoothing=0.3,
        leave=False,
    )

    # while true ?
    for count in range(
        int(dataloader._bulk_batch_count + dataloader._first_batch_count)
    ):
        # grab batch from internal dataloader queue
        current_batch = dataloader.grab_next_batch()
        if current_batch == "end_of_batch":
            break
        if current_batch == None:
            continue

        current_batch["input_ids"] = current_batch["input_ids"].reshape(
            -1, config_dict["text_encoder_context_window"]
        )
        current_batch["attention_mask"] = current_batch["attention_mask"].reshape(
            -1, config_dict["text_encoder_context_window"]
        )
        # just progress bar update
        train_step_progress_bar.update(1)

        unet_state, text_encoder_state, train_metric, train_rngs = train_step_funcs[
            current_batch["pixel_values"].shape
        ](
            # donated args
            unet_state,  # unet_state
            text_encoder_state,  # text_encoder_state
            # variable args
            current_batch,  # batch
            train_rngs,  # train_rng
            # unhashable static args
            frozen_vae,  # frozen_vae_state
            frozen_schedulers,
        )

        # TODO: save model function

    config_dict["chunk_number"] += 1

    save_dict_to_json(config_dict, config_dict_path)

# flush storage clean
for flushed_batch in range(
    config_dict["chunk_limit"] + config_dict["numb_of_prefetched_batch"] + 1
):
    dataloader.delete_prev_chunks(prev_chunk=flushed_batch)


print()
