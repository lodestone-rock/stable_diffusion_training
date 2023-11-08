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
    mu_quant_flag: chex.Array


def scale_by_lion_8bit(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_scale_dtype: Optional[chex.ArrayDType] = None,
    block_size: Optional[int] = 16,
    excluded_layer_mask: Optional[Any] = None,
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
        excluded_layer_mask: A tree with same structure as (or a prefix of) your params PyTree.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the quantization to, and `False` for those you want to skip. 

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

    def _block_quantize(leaf: chex.Array):
        # store original shape and pad for reconstruction later
        leaf_shape = leaf.shape
        # flatten >> reshape to [n,block_size] to increase MXU usage
        leaf = leaf.reshape(-1, block_size)
        # rescale the weight
        scales = jnp.max(jnp.abs(leaf), axis=-1, keepdims=True)
        # just in case the abs max scale is zero convert it to 1 to prevent zero division
        scales = 1 / jnp.where(
            scales <= min_norm, jnp.ones_like(scales, dtype=mu_scale_dtype), scales
        )
        leaf = leaf * scales
        # quantization happen after rescaling
        leaf = _quantize(leaf)
        return leaf, scales

    def _block_dequantize(
        leaf_shape: chex.Array,
        leaf: chex.Array,
        scales: chex.Array,
    ):
        # dequant before rescale
        leaf = _dequantize(leaf)
        # flatten then reshape it back to the original shape
        leaf = (leaf / scales).reshape(-1)
        leaf = leaf.reshape(leaf_shape.shape)
        return leaf 

    def _is_quantized(node):
        return isinstance(node, tuple)

    def _update_moment_quant(updates, moments, decay, order):
        """Compute the exponential moving average of the `order`-th moment."""
        param_shape = jax.tree_map(lambda x: jax.eval_shape(lambda y: y, x), updates)

        return jax.tree_util.tree_map(
            # https://github.com/google/jax/discussions/12826#discussioncomment-3894462
            # according to douglas first argument is used to infer tree structure so i dont have to do
            # anything special with state.mu_quant (a tuple)
            lambda g, t, shape: _block_quantize((1 - decay) * (g**order) + decay * _block_dequantize(shape, *t)) if _is_quantized(t) else (1 - decay) * (g**order) + decay * t,
            updates,  # the leaf is pure array
            moments,  # the leaf is a quantization params
            param_shape
        )

    def init_fn(params):
        # this one is the same shape as the param itself
        mu_quant = jax.tree_util.tree_map_with_path(  # moment
            # _block_quantize will quantize it if the flag is true, else it will stay the same
            lambda leaf, t, flag: _block_quantize(jnp.zeros_like(t, dtype=mu_scale_dtype)) if flag else jnp.zeros_like(t, dtype=mu_scale_dtype), 
            params, 
            excluded_layer_mask
        )
        return ScaleBy8bitLionState(count=jnp.zeros([], jnp.int32), mu_quant=mu_quant, mu_quant_flag=excluded_layer_mask)

    def update_fn(updates, state, params=None):
        del params
        param_shape = jax.tree_map(lambda x: jax.eval_shape(lambda y: y, x), updates)
        updates_new = jax.tree_util.tree_map(
            # https://github.com/google/jax/discussions/12826#discussioncomment-3894462
            # according to douglas first argument is used to infer tree structure so i dont have to do
            # anything special with state.mu_quant (a tuple)
            # _block_dequantize dequant the mu back and returning a tuple of quantized params
            lambda g, m, shape:jnp.sign((1.0 - b1) * g + b1 * _block_dequantize(shape, *m)) if _is_quantized(m) else jnp.sign((1.0 - b1) * g + b1 * m), 
            updates,
            state.mu_quant,
            param_shape
        )
        mu_quant = _update_moment_quant(updates, state.mu_quant, b2, 1)
        count_inc = numerics.safe_int32_increment(state.count)
        return updates_new, ScaleBy8bitLionState(count=count_inc, mu_quant=mu_quant, mu_quant_flag=state.mu_quant_flag)

    return base.GradientTransformation(init_fn, update_fn)


def lion_8bit(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_scale_dtype: Optional[Any] = None,
    block_size: int = 64,
    weight_decay: float = 1e-3,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    excluded_layer_mask: Optional[Any] = None,
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
        excluded_layer_mask: A tree with same structure as (or a prefix of) your params PyTree.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the quantization to, and `False` for those you want to skip. 
    Returns:
        The corresponding `GradientTransformation`.
    """
    return combine.chain(
        scale_by_lion_8bit(
            b1=b1, b2=b2, mu_scale_dtype=mu_scale_dtype, block_size=block_size, excluded_layer_mask=excluded_layer_mask
        ),
        transform.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate),
    )
