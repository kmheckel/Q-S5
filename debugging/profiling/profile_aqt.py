# %%
from aqt.jax.v2.aqt_dot_general import CalibrationMode
from functools import partial
from typing import Optional
import aqt.jax.v2.config as aqt_config
import jax.numpy as np
import jax

import timeit

# %%
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

# %%
def setup_inputs(key, H):
    return np.split(jax.random.normal(key, (2*H,)), 2)

def setup_inputs2d(key, H):
    return np.split(jax.random.normal(key, (2*H, H)), 2)

def had_quant(a, b):
    q_had(a, b).block_until_ready()

def had_float(a, b):
    np.multiply(a, b).block_until_ready()

def dot_quant(a, b):
    q_dot(a, b).block_until_ready()

def dot_float(a, b):
    np.dot(a, b).block_until_ready()

# %% [markdown]
# ## Real operations

# %%
H = 1000  # Set H to your desired value
key = jax.random.PRNGKey(0)  # Initialize the PRNG key
PRECISION = 8
q_had = q_had_maybe(precision=PRECISION)
q_dot = q_dot_maybe(precision=PRECISION)

# %%
print(f'Element-wise multiplication (Hadamard product) in {PRECISION}-bit with vectors ({H},)')
N, R = 500, 50
ftimes = timeit.repeat("had_float(a, b)", 
                      setup="a, b = setup_inputs(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
ftimes = np.array(ftimes) / N
print(f"Full precision execution time: {ftimes.mean()*1e6:5.2f}us +- {ftimes.std()*1e6:5.2f}us")
ftimes = timeit.repeat("had_float(a, b)", 
                      setup="a, b = setup_inputs(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
ftimes = np.array(ftimes) / N
print(f"Full precision execution time: {ftimes.mean()*1e6:5.2f}us +- {ftimes.std()*1e6:5.2f}us")

qtimes = timeit.repeat("had_quant(a, b)", 
                      setup="a, b = setup_inputs(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Quantized execution time:      {qtimes.mean()*1e6:5.2f}us +- {qtimes.std()*1e6:5.2f}us")
qtimes = timeit.repeat("had_quant(a, b)", 
                      setup="a, b = setup_inputs(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Quantized execution time:      {qtimes.mean()*1e6:5.2f}us +- {qtimes.std()*1e6:5.2f}us")

# %%
print(f'Matrix multiplication (dot product) in {PRECISION}-bit with matrices ({H}, {H})')
N, R = 100, 10
ftimes = timeit.repeat("dot_float(a, b)", 
                      setup="a, b = setup_inputs2d(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
ftimes = np.array(ftimes) / N
print(f"Full precision execution time: {ftimes.mean()*1e3:5.2f}ms +- {ftimes.std()*1e3:5.2f}ms")
ftimes = timeit.repeat("dot_float(a, b)", 
                      setup="a, b = setup_inputs2d(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
ftimes = np.array(ftimes) / N
print(f"Full precision execution time: {ftimes.mean()*1e3:5.2f}ms +- {ftimes.std()*1e3:5.2f}ms")

qtimes = timeit.repeat("dot_quant(a, b)", 
                      setup="a, b = setup_inputs2d(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Quantized execution time:      {qtimes.mean()*1e3:5.2f}ms +- {qtimes.std()*1e3:5.2f}ms")
qtimes = timeit.repeat("dot_quant(a, b)", 
                      setup="a, b = setup_inputs2d(key, H)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Quantized execution time:      {qtimes.mean()*1e3:5.2f}ms +- {qtimes.std()*1e3:5.2f}ms")

# %% [markdown]
# ## Complex operations

# %%



q_dot = q_dot_maybe(precision=PRECISION)

def setup_inputs_complex(key, H, matrix=False):
    shape = (2*H, H) if matrix else (2*H, )
    return np.split(jax.random.normal(key, shape, dtype=np.complex64), 2)

def dot_native(a, b):
    return np.dot(a, b).block_until_ready()

@jax.jit
def dot_manual_complex(a, b):
    return (np.dot(a.real, b.real) - np.dot(a.imag, b.imag) + 1j * (np.dot(a.real, b.imag) + np.dot(a.imag, b.real)))

def dot_manual(a, b):
    return dot_manual_complex(a, b).block_until_ready()

@jax.jit
def dot_manual_complex_quant(a, b):
    return (q_dot(a.real, b.real) - q_dot(a.imag, b.imag) + 1j * (q_dot(a.real, b.imag) + q_dot(a.imag, b.real)))

def dot_manual_quant(a, b):
    return dot_manual_complex_quant(a, b).block_until_ready()


H = 1000
print(f'Matrix multiplication (dot product) with complex matrices ({H}, {H})')
N, R = 100, 10
ftimes = timeit.repeat("dot_native(a, b)", 
                      setup="a, b = setup_inputs_complex(key, H, matrix=True)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
ftimes = np.array(ftimes) / N
print(f"Native execution time:       {ftimes.mean()*1e3:5.2f}ms +- {ftimes.std()*1e3:5.2f}ms")
ftimes = timeit.repeat("dot_native(a, b)", 
                      setup="a, b = setup_inputs_complex(key, H, matrix=True)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
ftimes = np.array(ftimes) / N
print(f"Native execution time:       {ftimes.mean()*1e3:5.2f}ms +- {ftimes.std()*1e3:5.2f}ms")

qtimes = timeit.repeat("dot_manual(a, b)", 
                      setup="a, b = setup_inputs_complex(key, H, matrix=True)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Manual execution time:       {qtimes.mean()*1e3:5.2f}ms +- {qtimes.std()*1e3:5.2f}ms")
qtimes = timeit.repeat("dot_manual(a, b)", 
                      setup="a, b = setup_inputs_complex(key, H, matrix=True)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Manual execution time:       {qtimes.mean()*1e3:5.2f}ms +- {qtimes.std()*1e3:5.2f}ms")

qtimes = timeit.repeat("dot_manual_quant(a, b)", 
                      setup="a, b = setup_inputs_complex(key, H, matrix=True)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Manual quant execution time: {qtimes.mean()*1e3:5.2f}ms +- {qtimes.std()*1e3:5.2f}ms")
qtimes = timeit.repeat("dot_manual_quant(a, b)", 
                      setup="a, b = setup_inputs_complex(key, H, matrix=True)", 
                      globals=globals(), 
                      number=N,
                      repeat=R)
qtimes = np.array(qtimes) / N
print(f"Manual quant execution time: {qtimes.mean()*1e3:5.2f}ms +- {qtimes.std()*1e3:5.2f}ms")