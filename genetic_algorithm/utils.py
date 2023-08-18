from typing import Optional

import importlib


class LazyLoader(dict):
    def __missing__(self, module_name: str):
        module = importlib.import_module(module_name)
        self[module_name] = module
        return module
