from training_utils import (
    TrainingConfig,
    on_device_model_training_state,
    dp_compile_all_unique_resolution
)
from streamer.dataloader import DataLoader
from streamer.utils import (
    numpy_to_pil_and_save,
    write_list_to_file,
    TimingContextManager,
    read_json_file,
)


# either use argparse or this
training_config = TrainingConfig(**read_json_file("model_properties.json"))

# some assertion check to check if the config is correct
assert len(training_config.image_area_root) == len(
    training_config.minimum_axis_length
), "number of elements in image_area_root and minimum_axis_length is not match! check your config files!"

# create model states to device
model_states = on_device_model_training_state(training_config)
# compile all posible resolution bucket
train_step_funcs = dp_compile_all_unique_resolution(*model_states, training_config)

print()
