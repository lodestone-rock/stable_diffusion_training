from typing import Any, Callable, Optional, Union
import jax.numpy as jnp
import jax
import chex
from typing import NamedTuple, Optional
import optax
from optax._src import transform, combine, numerics, base
from optax._src.alias import _scale_by_learning_rate, ScalarOrSchedule


# struct
class ScaleBy8bitLionState(NamedTuple):
    """State for the Lion algorithm."""

    count: chex.Array
    mu_quant: chex.Array  # quantized mu


def scale_by_lion_8bit(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_scale_dtype: Optional[chex.ArrayDType] = None,
    block_size: Optional[int] = 16,
    exclude_layer: list = [],
) -> base.GradientTransformation:
    """Rescale updates according to the Lion algorithm.

    References:z
        [Chen et al, 2023](https://arxiv.org/abs/2302.06675)

    Args:
        b1: Rate for combining the momentum and the current grad.
        b2: Decay rate for the exponentially weighted average of grads.
        mu_scale_dtype: Optional `dtype` to be used for the momentum; if
        `None` then the `dtype is inferred from `params` and `updates`.
        block_size: Optional `int` quantization block size.
        TODO: MASK!!!!
        TODO: MASK for unquantized param must be another dict to make it static

    Returns:
        A `GradientTransformation` object.
    """

    # just a helper just in case somone use "fp32" instead of jnp.float32
    mu_scale_dtype = jax.dtypes.canonicalize_dtype(mu_scale_dtype)
    # this offset ensures zero crossing
    offset = 3.7398995e-09
    min_norm = 0.0

    def _quantize(x: chex.Array):
        # extract the sign because the negative scale is the same
        q_sign = jnp.sign(x + offset)
        # square root of 5
        q = jnp.power(jnp.abs(x + offset), 1 / 5)
        # rescale to signed integer
        q = ((q * q_sign)) * 127
        return jnp.round(q).astype(jnp.int8)

    def _dequantize(q: chex.Array):
        # dequantize from integer back to float realm
        x = ((q / 127) ** 5).astype(mu_scale_dtype) - offset
        return x

    def _block_quantize(is_quantized: bool, leaf: chex.Array):
        if is_quantized:
            # TODO: perhaps remove this blocksize check if it's slow
            # padding by zero just in case it's not divisible by block size
            # pad the tail
            tail_pad = int(block_size - (leaf.size % block_size))
            # store original shape and pad for reconstruction later
            leaf_shape = leaf.shape
            # flatten >> pad >> reshape to [n,block_size]
            leaf = jnp.pad(leaf.reshape(-1), [0, tail_pad]).reshape(-1, block_size)

            # rescale the weight
            scales = jnp.max(jnp.abs(leaf), axis=-1, keepdims=True)
            # just in case the abs max scale is zero convert it to 1 to prevent zero division
            scales = jnp.where(
                scales <= min_norm, jnp.ones_like(scales, dtype=mu_scale_dtype), scales
            )
            leaf = leaf / scales

            # quantization happen after rescaling
            leaf = _quantize(leaf)
            # always pass the flag back
            return (is_quantized, leaf, scales, tail_pad, leaf_shape)
        else:
            # always pass the flag back
            return (is_quantized, leaf)

    def _block_dequantize(
        is_quantized: bool,
        leaf: chex.Array,
        scales: chex.Array = None,
        pad: int = None,
        leaf_shape=None,
    ):
        if is_quantized:
            # dequant before rescale
            leaf = _dequantize(leaf)
            # remove pad and reconstruct array back to the orig shape
            return (
                is_quantized,
                (leaf * scales).reshape(-1)[: leaf.size - pad].reshape(leaf_shape),
            )
        else:
            return (is_quantized, leaf)

    def _create_quantized_flag(leaf_name, param):
        """return a tuple where the first element a boolean true flag indicating if the param needs to be quantized"""
        # convert the dict name to key (im assuming the dict name is just a string here)
        layer_name_hiearch = tuple(key.key for key in leaf_name)

        is_included = True
        # if the layer is excluded from quantization just return the param itself
        if exclude_layer:
            for excluded in exclude_layer:
                if excluded in layer_name_hiearch:
                    included = False
                    break       
        return (is_included, param)

    def _update_moment_quant(updates, moments, decay, order):
        """Compute the exponential moving average of the `order`-th moment."""
        return jax.tree_util.tree_map(
            # https://github.com/google/jax/discussions/12826#discussioncomment-3894462
            # according to douglas first argument is used to infer tree structure so i dont have to do
            # anything special with state.mu_quant (a tuple)
            lambda g, t: _block_quantize(
                t[0],  # put the flag back
                (1 - decay) * (g**order)
                + decay * _block_dequantize(*t)[1],  # dequant then calculate grad
            ),
            updates,  # the leaf is pure array
            moments,  # the leaf is a tuple with quantization flag as the first element
        )

    def init_fn(params):
        # this one is the same shape as the param itself
        mu_quant = jax.tree_util.tree_map_with_path(  # moment
            # _create_quantized_flag returns a tuple of params with quantization flag for the first element
            # _block_quantize will quantize it if the flag is true, else it will stay the same
            lambda leaf, t: _block_quantize(
                *_create_quantized_flag(leaf, jnp.zeros_like(t, dtype=mu_scale_dtype))
            ),
            params,
        )

        # make sure initialization for the scale array is not zero lol
        # or else you're gonna divide by zero and you're gonna have a bad day!
        # also the shape is not identical to the params because it's a slice of the params itself
        # some layer must get excluded from this
        # TODO: MASK!!!

        return ScaleBy8bitLionState(count=jnp.zeros([], jnp.int32), mu_quant=mu_quant)

    def update_fn(updates, state, params=None):
        del params
        updates_new = jax.tree_util.tree_map(
            # https://github.com/google/jax/discussions/12826#discussioncomment-3894462
            # according to douglas first argument is used to infer tree structure so i dont have to do
            # anything special with state.mu_quant (a tuple)
            # _block_dequantize dequant the mu back and returning a tuple with flag and params
            # remove the flag then compute the gradient as usual
            lambda g, m: jnp.sign((1.0 - b1) * g + b1 * _block_dequantize(*m)[1]),
            updates,
            state.mu_quant,
        )
        mu_quant = _update_moment_quant(updates, state.mu_quant, b2, 1)
        # no casting!
        # mu = utils.cast_tree(mu, mu_scale_dtype)
        count_inc = numerics.safe_int32_increment(state.count)
        return updates_new, ScaleBy8bitLionState(count=count_inc, mu_quant=mu_quant)

    return base.GradientTransformation(init_fn, update_fn)


def lion_8bit(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_scale_dtype: Optional[Any] = None,
    block_size: int = 64,
    weight_decay: float = 1e-3,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    exclude_layer: Optional[list] = None,
) -> base.GradientTransformation:
    """The Lion optimizer.
    Lion is discovered by symbolic program search. Unlike most adaptive optimizers
    such as AdamW, Lion only tracks momentum, making it more memory-efficient.
    The update of Lion is produced through the sign operation, resulting in a
    larger norm compared to updates produced by other optimizers such as SGD and
    AdamW. A suitable learning rate for Lion is typically 3-10x smaller than that
    for AdamW, the weight decay for Lion should be in turn 3-10x larger than that
    for AdamW to maintain a similar strength (lr * wd).
    References:
        Chen et al, 2023: https://arxiv.org/abs/2302.06675
    Args:
        learning_rate: A fixed global scaling factor.
        b1: Rate to combine the momentum and the current gradient.
        b2: Exponential decay rate to track the momentum of past gradients.
        mu_scale_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay: Strength of the weight decay regularization. Note that this
        weight decay is multiplied with the learning rate. This is consistent
        with other frameworks such as PyTorch, but different from
        (Loshchilov et al, 2019) where the weight decay is only multiplied with
        the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the weight decay to, and `False` for those you want to skip. Note
        that the Adam gradient transformations are applied to all parameters.
        exclude_layer: a list of layer names that's excluded from the quantization.
        # TODO: make the input same ask mask argument instead of list!  
    Returns:
        The corresponding `GradientTransformation`.
    """
    return combine.chain(
        scale_by_lion_8bit(
            b1=b1, b2=b2, mu_scale_dtype=mu_scale_dtype, block_size=block_size, exclude_layer=exclude_layer
        ),
        transform.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate),
    )
