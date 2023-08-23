from typing import Callable
import functools
import importlib
import inspect
import numpy as np


def is_int(value):
    return isinstance(value, (int, np.integer))


def is_real_valued_array(array):
    return np.issubdtype(array.dtype, np.floating)


def kwargs_proxy(func: Callable) -> Callable:
    """
    A decorator that filters keyword arguments based on the original function's signature.

    This decorator takes a callable function and dynamically filters out invalid
    keyword arguments from the provided keyword arguments based on the original
    function's parameter names. It ensures that only valid keyword arguments are
    passed to the wrapped function.

    Parameters:
        func (Callable): The original callable function to be wrapped.

    Returns:
        Callable: A wrapped function that filters keyword arguments.

    Example Usage:
        ```python

        @kwargs_proxy
        def my_function(a: int, b: int, *, x: int, y: int):
            # Function implementation

        ```

    Note:
        This function uses the `inspect` module to extract the keyword arguments
        from the original function's signature and filters out any invalid
        keyword arguments.
    """
    valid_kwargs = set(inspect.signature(func).parameters)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = {kwarg: value for kwarg, value in kwargs.items()
                  if kwarg in valid_kwargs}

        return func(*args, **kwargs)

    return wrapper


class LazyLoader(dict):
    """
    A dictionary-based lazy loader for importing modules. We use this to avoid
    circular imports.

    This class is designed to allow lazy loading of modules. When an attempt is
    made to access a module that hasn't been imported yet, the `__missing__`
    method is triggered, and the requested module is dynamically imported using
    the `importlib` module. The imported module is then stored in the loader's
    dictionary for future reference.

    Example Usage:
        ```python

        loader = LazyLoader()  # no modules loaded yet
        my_module = loader['my_module']  # my_module is loaded here

        ```

    Note:
        This class inherits from the built-in `dict` class and adds lazy loading
        functionality.
    """

    def __missing__(self, module_name: str):
        """
        Import a missing module lazily.

        This overrides the default __missing__ method of the dict class

        Parameters:
            module_name (str): The name of the module to be imported.

        Returns:
            module: The imported module.
        """
        module = importlib.import_module(module_name)
        self[module_name] = module
        return module
