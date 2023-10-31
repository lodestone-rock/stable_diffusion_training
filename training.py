import time
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from training_utils import (
    TrainingConfig,
    on_device_model_training_state,
    dp_compile_all_unique_resolution,
    save_model,
)
from streamer.dataloader import DataLoader
from streamer.utils import (
    TimingContextManager,
    read_json_file,
    save_dict_to_json,
    delete_file_or_folder,
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
def main():
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
        unet_ema,
        text_encoder_ema,
        frozen_vae,
        frozen_schedulers,
        model_object_dict,
    ) = on_device_model_training_state(training_config)

    # compile all posible resolution bucket
    train_step_funcs = dp_compile_all_unique_resolution(
        unet_state, text_encoder_state, unet_ema, text_encoder_ema, frozen_vae, frozen_schedulers, training_config
    )

    if config_dict["DEBUG"]:
        # carefull when doing this, it will overwrite the json states!
        config_dict["loss_logging_interval"] //= 10
    if not os.path.isfile(config_dict["loss_csv"]):
        with open(config_dict["loss_csv"], "w") as loss_file:
            loss_file.write("steps, step_size, loss, time, chunk, seed\n")

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
        )
        dataloader.prepare_training_dataframe()
        dataloader.create_training_dataframe()
        if config_dict["DEBUG"]:
            dataloader._bulk_batch_count = 100  # debug limit to 100 batch
        dataloader.dispatch_worker()

        # progress bar
        train_step_progress_bar = tqdm(
            total=int(dataloader._bulk_batch_count + dataloader._first_batch_count),
            desc="Training...",
            position=1,
            smoothing=0.3,
            leave=False,
        )

        # test if saving model works so you dont waste compute!
        try:
            print("trying to save model to check if the saving mechanism works")
            save_model(
                model_object_dict=model_object_dict,
                tokenizer_object=tokenizer,
                unet_params=unet_state.params,
                text_encoder_params=text_encoder_state.params,
                vae_params=frozen_vae.params,
                output_dir=config_dict["test_save_path"],
            )
            if unet_ema is not None and text_encoder_ema is not None:
                save_model(
                    model_object_dict=model_object_dict,
                    tokenizer_object=tokenizer,
                    unet_params=unet_ema,
                    text_encoder_params=text_encoder_ema,
                    vae_params=frozen_vae.params,
                    output_dir=config_dict["test_save_path"],
                )
        except Exception as e:
            print(
                f"failed to save model prior to training session! please check your config or your code first"
            )
            print(f"reason: {e}")
            sys.exit()

        print("save function works as expected deleting the test model")
        # delete it afterwards because it's not needed
        delete_file_or_folder(config_dict["test_save_path"])

        start = time.time()

        # training loop
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
            train_step_progress_bar.set_description(
                desc=f'processing shape [{current_batch["pixel_values"].shape}]...'
            )

            # store loss value
            train_metrics = []

            unet_state, text_encoder_state, unet_ema, text_encoder_ema, train_metric, train_rngs = train_step_funcs[
                current_batch["pixel_values"].shape
            ](
                # donated args
                unet_state,  # unet_state
                text_encoder_state,  # text_encoder_state
                unet_ema,
                text_encoder_ema,
                # variable args
                current_batch,  # batch
                train_rngs,  # train_rng
                # unhashable static args
                frozen_vae,  # frozen_vae_state
                frozen_schedulers,
                # static args not necessary here
            )

            # this appending method wont cause jax to wait until ready
            # it will just return jax.array futures and proceed further
            train_metrics.append(train_metric["loss"])

            # just storing loss value
            if count % config_dict["loss_logging_interval"] == 0:
                stop = time.time()
                time_elapsed = round(stop - start, 4)
                # force jax to return all promised values
                loss = sum(train_metrics) / len(train_metrics)
                time_per_step = round(
                    time_elapsed / config_dict["loss_logging_interval"], 4
                )
                start = time.time()
                train_step_progress_bar.write(
                    f'at steps {count}, avg loss for {config_dict["loss_logging_interval"]} steps: {loss},'
                    f"took {time_elapsed} second(s) or {time_per_step} second(s) per step"
                )
                with open(config_dict["loss_csv"], "a") as loss_file:
                    # step_size, loss, time, chunk, seed
                    loss_file.write(
                        f'\n{count},{config_dict["loss_logging_interval"]},{loss},{time_elapsed},{config_dict["chunk_steps"]},{config_dict["master_seed"]}'
                    )

        # latest model folder path just in case something bad happened

        model_path_without_chunk_number = config_dict["model_path"].split("@")[0]
        latest_model_path = (
            f'{model_path_without_chunk_number}@{config_dict["chunk_steps"]}'
        )
        save_model(
            model_object_dict=model_object_dict,
            tokenizer_object=tokenizer,
            unet_params=unet_state.params,
            text_encoder_params=text_encoder_state.params,
            vae_params=frozen_vae.params,
            output_dir=latest_model_path,
        )
        if unet_ema is not None and text_encoder_ema is not None:
            save_model(
                model_object_dict=model_object_dict,
                tokenizer_object=tokenizer,
                unet_params=unet_ema,
                text_encoder_params=text_encoder_ema,
                vae_params=frozen_vae.params,
                output_dir=f"{latest_model_path}-EMA",
            )
        # only save n latest chunk so it not cluttering the storage
        delete_file_or_folder(
            f'{model_path_without_chunk_number}@{config_dict["chunk_steps"]-config_dict["keep_trained_model_buffer"]}'
        )

        # update states in json
        config_dict["model_path"] = latest_model_path
        config_dict["chunk_number"] += 1
        config_dict["chunk_steps"] += 1

        save_dict_to_json(config_dict, config_dict_path)

    # flush storage clean
    for flushed_batch in range(
        config_dict["chunk_limit"] + config_dict["numb_of_prefetched_batch"] + 1
    ):
        dataloader.delete_prev_chunks(prev_chunk=flushed_batch)

    config_dict["master_seed"] += 1
    save_dict_to_json(config_dict, config_dict_path)


if __name__ == "__main__":
    main()
print()
