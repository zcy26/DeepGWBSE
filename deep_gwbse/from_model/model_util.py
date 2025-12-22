import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as e2nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from functools import wraps
import logging
import inspect
import time
import functools
import tracemalloc

eV2Ry = 0.073498618
Ry2eV = 13.6056980659


def print_model_size(model, model_name="Model"):
    param_size = 0
    param_number = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_number += param.numel()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{model_name} parameters: {param_number} with size of {size_all_mb:.3f} MB')
    return param_number

def timeCudaWatch(func):
    """Decorator to measure execution time (GPU if available, else CPU)."""
    @wraps(func)
    def wrapper(*args, **kwargs):

        # GPU timing path
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            result = func(*args, **kwargs)
            end.record()

            torch.cuda.synchronize()
            logging.debug(
                f"{func.__name__} execution time: {start.elapsed_time(end) * 1e-3:.3f} s"
            )
            return result

        # CPU timing path
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()

        logging.debug(f"{func.__name__} execution time: {t1 - t0:.3f} s")
        return result

    return wrapper





class H5ls:
    def __init__(self):
        # Store an empty list for dataset names
        self.names = []

    def __call__(self, name, h5obj):
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(h5obj,'dtype') and not name in self.names:
            self.names += [name]


def time_watch(func):
    # TODO: extened to MPI4PY
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.debug(f"{func.__name__} time used: {end_time - start_time:.6f} seconds")
        return result
    return wrapper


def memory_watch(top_n=None):
    # TODO: extened to MPI4PY
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            result = func(*args, **kwargs)
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            # print(f"[ Top {top_n} Memory]")
            total = 0
            for stat in top_stats:
                total += stat.size / 1024 / 1024 / 1024
            print(f"\n{func.__name__} memory used: {total:.2f} GB")
            return result
        return wrapper
    return decorator


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def capture_config(init):
    """
    This only works for __init__ methods which takes int, float, str, list, dict as arguments. (doesn't support obj such as Transformer)
    """
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        # Save all args except 'self'
        self.model_config = {k: v for k, v in bound.arguments.items() if k != 'self'}
        self.model_config = convert_to_serializable(self.model_config)
        return init(self, *args, **kwargs)
    return wrapper

