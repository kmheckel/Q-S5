from aqt.jax.v2.aqt_dot_general import CalibrationMode
from functools import partial
from typing import Optional
import aqt.jax.v2.config as aqt_config
import jax.numpy as np
import jax

import timeit

fully_quantized = partial(
    aqt_config.fully_quantized,
    calibration_mode=CalibrationMode.ALL_AXES, use_stochastic_rounding=False,
)

def q_dot_maybe(precision: Optional[int]):
    if precision is None:
        return np.dot
    else:
        dot_general = fully_quantized(fwd_bits=precision, bwd_bits=precision)
        return quant_dot_for_dot(dot_general)

def q_had_maybe(precision: Optional[int]):
    if precision is None:
        return np.multiply
    else:
        dot_general = fully_quantized(fwd_bits=precision, bwd_bits=precision)
        return quant_dot_for_hadamard(dot_general)

def quant_dot_for_hadamard(dot_general):
    """Generate a jitted general_dot function to be used for hadamard products.
    Note that this function does not support batch dimensions. All dimensions will
    be used for calibration in the quantization."""
    def _dot(a, b):
        contr_dims = ((), ())  # hadamard has no contracting dims
        batch_dims = (tuple(range(a.ndim)), tuple(range(b.ndim)))  # use all dims as batch dims
        return dot_general(a, b, (contr_dims, batch_dims))
    return jax.jit(_dot)

def quant_dot_for_dot(general_dot):
    """Generate a jitted general_dot function to be used for dot products.
    Will contract on the last dimension of a, and the first dimension of b.
    This means that there are no batch dimensions, and all dimensions will be used
    for calibration in the quantization."""
    def _dot(a, b):
        # contr_dims = ((a.ndim-1,), (1,))  # batched version (not used)
        # batch_dims = ((0,), (0,))  # batched version (not used)
        contr_dims = ((a.ndim-1,), (0,))
        batch_dims = ((), ())
        return general_dot(a, b, (contr_dims, batch_dims))
    return jax.jit(_dot)

qh = q_had_maybe(4)
qd = q_dot_maybe(4)
key = jax.random.PRNGKey(0)
H = 100
a, b = np.split(jax.random.normal(key, (2*H,)), 2)
qh(a, b)
a, b = np.split(jax.random.normal(key, (2*H, H)), 2)
qd(a, b)