import json
import gc
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Any

from training_utils import (
    TrainingConfig, 
    calculate_resolution_array, 
    load_models,
    create_lion_optimizer_states,
    create_frozen_states
)
from streamer.dataloader import DataLoader
from streamer.utils import (
    numpy_to_pil_and_save,
    write_list_to_file,
    TimingContextManager,
    read_json_file,
)

#sharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

rng = jax.random.PRNGKey # create alias because i cant remember this shit!
# adjust this sharding mesh to create appropriate sharding rule
# assume we have 8 device
# (1,8) = model parallel
# (8,1) = data parallel
# (4,2)/(2,4) = model data parallel
devices = mesh_utils.create_device_mesh((jax.device_count(),1))
# create axis name on how many parallelism slice you want on your model
mesh = Mesh(devices, axis_names=('data_parallel', 'model_parallel')) 

# either use argparse or this
training_config = TrainingConfig(**read_json_file("model_properties.json"))
models = load_models(model_dir=training_config.model_path)



# i haven't implemented disabling either model params during traing!
trained_model_states = create_lion_optimizer_states(
    models=models,
    train_text_encoder=True,
    train_unet=True,
    adam_to_lion_scale_factor=7
)

frozen_states = create_frozen_states(
    models=models,
)


# you can't pass this as a variable since this will get traced by jax
# but this method is needed for loss computation so i explictly defined it here
# i shouldve wrap the entire thing inside a class X_X
# https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html
# TODO: use pytreenode so it can be used in function transformation!!!
# vae_fn=models["vae"]["vae_model"]
# noise_scheduler_fn=models["schedulers"]["noise_scheduler_object"]

# TODO: this train step function is not finished!
def train_step(
    # donated args
    unet_state:Any, # define sharding rule!
    text_encoder_state:Any, # define sharding rule!
    # variable args
    batch:dict, # define sharding rule!
    train_rng:jax.random.PRNGKey, # define sharding rule!
    frozen_vae_state: Any,
    frozen_noise_scheduler_state: Any, # welp technically not a trainable by any means
    # unhashable static args
    # noise_scheduler_state:Any, # unhashable
    # vae_params:dict, # unhashable
    # vae_fn:Any, # unhashable
    # noise_scheduler_fn:Any, # unhashable
    use_offset_noise:bool=False,
    strip_bos_eos_token:bool=True
):

    """
    this jittable trainstep function just lightly wraps 
    the actual loss function and adding some states to it
    """

    # generate rng and return new_train_rng to be used for the next iteration step
    # rng is comunicated though device aparently
    dropout_rng, sample_rng, new_train_rng = jax.random.split(
        train_rng, num=3)

    def compute_loss(
        unet_params, 
        text_encoder_params,
        vae_params,
        noise_scheduler_state,
        batch
        ):
        # Convert images to latent space
        vae_outputs = frozen_vae_state.call.apply(
            variables={"params": frozen_vae_state.params},
            sample=batch["pixel_values"],
            deterministic=True,
            method=frozen_vae_state.call.encode
        )

        # get sample distribution from VAE latent
        latents = vae_outputs.latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        # weird scaling don't touch it's a lazy normalization
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        # I think I should combine this with the first noise seed generator
        noise_offset_rng, noise_rng, timestep_rng = jax.random.split(
            key=sample_rng, num=3)
        noise = jax.random.normal(key=noise_rng, shape=latents.shape)
        if use_offset_noise:
            # mean offset noise, why add offset?
            # here https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise_offset = jax.random.normal(
                key=noise_offset_rng,
                shape=(latents.shape[0], latents.shape[1], 1, 1)
            ) * 0.1
            noise = noise + noise_offset

        # Sample a random timestep for each image
        batch_size = latents.shape[0]
        timesteps = jax.random.randint(
            key=timestep_rng,
            shape=(batch_size,),
            minval=0,
            maxval=frozen_noise_scheduler_state.call.config.num_train_timesteps,
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = frozen_noise_scheduler_state.call.add_noise(
            state=frozen_noise_scheduler_state.params,
            original_samples=latents,
            noise=noise,
            timesteps=timesteps
        )
        print(batch["input_ids"].shape)
        encoder_hidden_states = text_encoder_state.apply_fn(
            params=text_encoder_params,
            input_ids=batch["input_ids"],
            dropout_rng=dropout_rng,
            train=True
        )[0]
        print(encoder_hidden_states.shape)
        # reshape encoder_hidden_states to shape (batch, token_append, token, hidden_states)
        encoder_hidden_states = jnp.reshape(
            encoder_hidden_states,
            (latents.shape[0], -1, 77, encoder_hidden_states.shape[-1]),
        )
        print(encoder_hidden_states.shape)

        if strip_bos_eos_token:
            encoder_hidden_states = jnp.concatenate(
                [
                    # first encoder hidden states without eos token
                    encoder_hidden_states[:, 0, :-1, :],
                    # the rest of encoder hidden states without both bos and eos token
                    jnp.reshape(
                        encoder_hidden_states[:, 1:-1, 1:-1, :],
                        (
                            encoder_hidden_states.shape[0],
                            -1,
                            encoder_hidden_states.shape[-1]
                        )
                    ),
                    # last encoder hidden states without bos token
                    encoder_hidden_states[:, -1, 1:, :]
                ],
                axis=1
            )
        else:
            # reshape encoder_hidden_states to shape (batch, token_append & token, hidden_states)
            encoder_hidden_states = jnp.reshape(
                encoder_hidden_states,
                (encoder_hidden_states.shape[0], -
                    1, encoder_hidden_states.shape[-1])
            )
        print(encoder_hidden_states.shape)

        # Predict the noise residual because predicting image is hard :P
        # essentially try to undo the noise process
        model_pred = unet_state.apply_fn(
            variables={"params": unet_params},
            sample=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            train=True
        ).sample

        # Get the target for loss depending on the prediction type
        # sd1.x use epsilon aka noise residual but sd2.1 use velocity prediction
        if frozen_noise_scheduler_state.call.config.prediction_type == "epsilon":
            target = noise
        elif frozen_noise_scheduler_state.call.config.prediction_type == "v_prediction":
            target = frozen_noise_scheduler_state.call.get_velocity(
                state=frozen_noise_scheduler_state.params,
                sample=latents,
                noise=noise,
                timesteps=timesteps
            )
        else:
            # panic!!
            raise ValueError(
                f"Unknown prediction type {frozen_noise_scheduler_state.call.config.prediction_type}")

        # MSE loss
        loss = (target - model_pred) ** 2
        loss = loss.mean()

        return loss

    # perform autograd
    # TODO: define the differentiable input ! 
    # i havent updated this to include all params!

    # autograd transform function to get gradient of the input params
    # TODO: use reduce_axes to sum all of the gradient inplace! 
    # this will significantly reduce memory consumption
    grad_fn = jax.value_and_grad(
        fun=compute_loss,
        argnums=[0,1] # differentiate first and second input 
    )
    # grad is a tuple here because multiple params is provided
    loss, grad = grad_fn(
        unet_state.params, # unet_params
        text_encoder_state.params, # text_encoder_params
        frozen_vae_state.params, # frozen_vae_state.params
        frozen_noise_scheduler_state.params, # frozen_noise_scheduler_state.params
        batch, # batch
        )

    # update weight and bias value
    new_unet_state = unet_state.apply_gradients(grads=grad[0])
    new_text_encoder_state = text_encoder_state.apply_gradients(
        grads=grad[1])

    # calculate loss
    metrics = {"loss": loss}

    # idk how jax check this output for donation 
    # but just in case i put the donated args with the same position as the input
    # donated args are new_unet_state and new_text_encoder_state since it has the same 
    # data structure so inplace update is good
    return new_unet_state, new_text_encoder_state, metrics, new_train_rng


jax.profiler.start_trace("./tensorboard")

#TODO: put this RNG elsewere
train_rngs = rng(2)

# this is ugly but eh it works
# can always override the dictionary but eh
unet_state = jax.tree_map(
    lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), 
    trained_model_states["unet_state"], 
)
text_encoder_state = jax.tree_map(
    lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), 
    trained_model_states["text_encoder_state"], 
)
frozen_vae = jax.tree_map(
    lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), 
    frozen_states["vae_state"], 
)
frozen_schedulers = jax.tree_map(
    lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), 
    frozen_states["schedulers_state"], 
)


dummy_batch_size = 8*4
# dummy_batch
with jax.default_device(jax.devices("cpu")[0]):
    batch = {
        'attention_mask': jnp.arange(dummy_batch_size * 3 * 77).reshape(dummy_batch_size, 3, 77), 
        'input_ids': jnp.arange(dummy_batch_size * 3 * 77).reshape(dummy_batch_size * 3, 77), 
        'pixel_values': jax.random.uniform(train_rngs, shape=(dummy_batch_size , 3, 512, 512))
    }
# define sharding rule (im doing data parallelism here)
batch = jax.tree_map(
    lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec("data_parallel", None))), 
    batch, 
)


# just gonna be verbose here for less headache
p_train_step = jax.jit(
    train_step, 
    # donated arguments (inplace update)
    donate_argnums=(
        0, # "unet_state"
        1, # "text_encoder_state"
    ), 
    in_shardings=(
        # unet_state
        jax.tree_map(
            lambda leaf: NamedSharding(mesh, PartitionSpec()), 
            trained_model_states["unet_state"], 
        ),
        # text_encoder_state
        jax.tree_map(
            lambda leaf: NamedSharding(mesh, PartitionSpec()), 
            trained_model_states["text_encoder_state"], 
        ),
        # batch
        jax.tree_map(
            lambda leaf: NamedSharding(mesh, PartitionSpec("data_parallel", None)), 
            batch, 
        ),
        # rngs
        None, # honestly, donno how to shard this one, COMPILER! TAKE THE WHEEL HERE
        # frozen_vae
        jax.tree_map(
            lambda leaf: NamedSharding(mesh, PartitionSpec()), 
            frozen_states["vae_state"], 
        ),
        # frozen_schedulers
        jax.tree_map(
            lambda leaf: NamedSharding(mesh, PartitionSpec()), 
            frozen_states["schedulers_state"], 
        ),
        # use_offset_noise
        # None, # moved to static
        # strip_bos_eos_token
        # None, # moved to static
    ),
    # compiled as static value
    # only hashable one!
    static_argnames=(
        "use_offset_noise",
        "strip_bos_eos_token",
    ),
    out_shardings=(
        jax.tree_map(
            lambda leaf: NamedSharding(mesh, PartitionSpec()), 
            trained_model_states["unet_state"], 
        ),
        jax.tree_map(
            lambda leaf: NamedSharding(mesh, PartitionSpec()), 
            trained_model_states["text_encoder_state"], 
        ),
        {"loss":  NamedSharding(mesh, PartitionSpec())},
        None,
    ),

)
del trained_model_states, frozen_states
unet_state, text_encoder_state, metrics, train_rngs = p_train_step(
    # donated args
    unet_state, # unet_state
    text_encoder_state, # text_encoder_state
    # variable args
    batch, # batch
    train_rngs, # train_rng
    # unhashable static args
    frozen_vae, # frozen_vae_state
    frozen_schedulers, # frozen_noise_scheduler_state
    # noise_scheduler_state=models["schedulers"]["noise_scheduler_state"],
    # vae_params=models["vae"]["vae_params"],
    # static args
    False, # use_offset_noise
    True, # strip_bos_eos_token
)

jax.profiler.stop_trace()
print()
