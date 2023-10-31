##### code mod start here #####
from typing import Any, Callable, Optional, Union
import jax.numpy as jnp
import jax
import chex
from typing import NamedTuple, Optional
import optax
from optax._src import transform, combine, numerics, base
from optax._src.alias import _scale_by_learning_rate, ScalarOrSchedule
from optax._src.transform import update_moment

class ScaleBy8bitLionState(NamedTuple):
    """State for the Lion algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu_qtree: chex.Array

def scale_by_8bit_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    blk_size: int = 16,
    min_norm: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Rescale updates according to the Lion algorithm.

    References:
      [Chen et al, 2023](https://arxiv.org/abs/2302.06675)
      [Dettmers et al, 2023](https://arxiv.org/abs/2110.02861)

    Args:
      b1: Rate for combining the momentum and the current grad.
      b2: Decay rate for the exponentially weighted average of grads.
      mu_dtype: Optional `dtype` to be used for the scale of quantized
      momentum; if `None` then the `dtype is inferred from `params` and
      `updates`.
    Returns:
      A `GradientTransformation` object.
    """

    def qt_leaf(node):
        return (isinstance(node, tuple) and len(node) == 2)

    def _canonicalize(dtype):
        if dtype is not None:
            return jax.dtypes.canonicalize_dtype(dtype)

    mu_dtype = _canonicalize(mu_dtype)

    def _pad_len(params):
        length = sum(a.size for a in jax.tree_util.tree_leaves(params))
        modulo = length % blk_size
        if modulo != 0:
            return blk_size - modulo
        else:
            return 0

    # DO NOT DARE TO TOUCH ANYTHING IN THIS

    def _dtree_quantize(x):
        q_sign = jnp.sign(x + 3.7398995e-09)
        q = jnp.power(jnp.abs(x + 3.7398995e-09), 1/5)
        q = ((q * q_sign)) * 127
        return jnp.round(q).astype(jnp.int8)

    def _dtree_dequantize(q):
        x = ((q / 127) ** 5).astype(mu_dtype) - 3.7398995e-09
        return x

    def _flatten(params):
        params = jax.tree_util.tree_leaves(params)
        params = jnp.concatenate([a.reshape(-1) for a in params], dtype=mu_dtype)
        return params

    def _quantize(values: chex.Array):
        scales = jnp.max(jnp.abs(values), axis=-1, keepdims=True)
        scales = 1 / jnp.where(scales <= min_norm, jnp.ones_like(scales, dtype=mu_dtype), scales)
        values = _dtree_quantize(values * scales)
        return values, scales

    def _block_quantize(kp, values):
        if (kp[-1].key in ['embedding', 'bias', 'scale']) or (kp[-2].key in ['conv_in', 'conv_out']) or (kp[-3].key == 'time_embedding'):
            return values.astype(mu_dtype)
        padding = [(0, _pad_len(values))]
        values = _flatten(values)
        values = jnp.pad(values, padding)
        values = values.reshape(-1, blk_size)
        return _quantize(values)

    def _block_dequantize(node, pad_len):
        if not qt_leaf(node):
            return node
        values, scales = node
        values = _dtree_dequantize(values)
        values = values.astype(scales.dtype) / scales
        values = values.reshape(-1)
        if pad_len != 0:
            values = values[:-pad_len]
        return values

    def _unflatten(updates, flat):
        """Extracts tensors from flat, using the structure and shapes of params."""
        updates_flat, treedef = jax.tree_util.tree_flatten(updates)
        offsets = []
        for update in updates_flat:
            size = update.size
            if offsets:
                offsets.append(size + offsets[-1])
            else:
                offsets.append(size)
        del offsets[-1]
        flat_split = jnp.split(flat, offsets)
        reshaped = [
            jnp.reshape(flat_update, update.shape)
            for flat_update, update in zip(flat_split, updates_flat)
        ]
        return jax.tree_util.tree_unflatten(treedef, reshaped)

    def init_fn(params):
        mu_qtree = jax.tree_util.tree_map_with_path(
            _block_quantize, params
        )
        mu_qtree = jax.tree_map(
            lambda x: (jnp.full(x[0].shape, 0, dtype=jnp.int8), jnp.ones(x[1].shape, dtype=mu_dtype)) if qt_leaf(x) 
            else jnp.ones(x.shape, dtype=mu_dtype), mu_qtree, is_leaf=qt_leaf
        )

        return ScaleBy8bitLionState(
            count=jnp.zeros([], jnp.int32),
            mu_qtree=mu_qtree
        )

    def update_fn(updates, state, params=None):
        del params
        param_shape = jax.tree_map(lambda x: jax.eval_shape(lambda y: y, x), updates)
        mu = jax.tree_map(lambda x, y: _unflatten(y, _block_dequantize(x, _pad_len(y))),
                           state.mu_qtree, param_shape, is_leaf=qt_leaf)
        
        updates_new = jax.tree_util.tree_map(
            lambda g, m: jnp.sign((1. - b1) * g + b1 * m), updates, mu)
        mu = update_moment(updates, mu, b2, 1)
        mu_qtree = jax.tree_util.tree_map_with_path(_block_quantize, mu)
        count_inc = numerics.safe_int32_increment(state.count)
        
        return updates_new, ScaleBy8bitLionState(
            count=count_inc, mu_qtree=mu_qtree
        )

    return base.GradientTransformation(init_fn, update_fn)

optax._src.transform.scale_by_8bit_lion = scale_by_8bit_lion

def lion_8bit(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    blk_size: int = 64,
    weight_decay: float = 1e-3,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
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
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
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

    Returns:
        The corresponding `GradientTransformation`.
    """
    return combine.chain(
        transform.scale_by_8bit_lion(b1=b1, b2=b2, mu_dtype=mu_dtype, blk_size=blk_size),
        transform.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate),
    )
##### code mod end here #####