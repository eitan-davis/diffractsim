import numpy
import numba
numba.jit
"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


global JAX_AVAILABLE
global CUPY_CUDA_AVAILABLE

try:
    import cupy
    CUPY_CUDA_AVAILABLE = True
except ImportError:
    CUPY_CUDA_AVAILABLE = False



try:
    import jax.numpy
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False




global backend
backend = numpy
global backend_name
backend_name = 'numpy'

def cpu_accelerator(function, accelerator_name = 'numba', *args, **options):
    """
    This function is a decorattor on all the calculating operation 
    to accelarate the work on the CPU with paralle prosesing 
    when relavent (the CPU is in use).
    """
    #if cupy is in use then CPU accelerator will interfere with it. 
    if backend == cupy:
        return function
    if accelerator_name == "numba":
        numba.jit(function, *args, **options)
    else:
        return function

def set_backend(name: str):
    """ Set the backend for the simulations
    This way, all methods of the backend object will be replaced.
    Args:
        name: name of the backend. Allowed backend names:
            - ``CPU``
            - ``CUDA``
            - ``JAX``
    """
    # perform checks
    if name == "CUDA" and not CUPY_CUDA_AVAILABLE:
        raise RuntimeError(
            "Cupy CUDA backend is not available.\n"
            "Do you have a GPU on your computer?\n"
            "Is Cupy with CUDA support installed?"
        )
    global backend
    global backend_name

    # change backend
    if name == "CPU":
        backend = numpy
        backend_name = 'numpy'

    elif name == "CUDA":
        backend = cupy
        backend_name = 'cupy'

    elif name == "JAX":
        backend = jax.numpy
        backend_name = 'jax'
    else:
        raise RuntimeError(f'unknown backend "{name}"')


def get_backend():
    global backend    
    print(backend)




