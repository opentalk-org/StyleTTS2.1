"""Preload native libraries without changing the process library search path.

The development shell places this module beside a generated preload.json.
Libraries are loaded by absolute path so later wheel dependencies resolve by
SONAME from the process link map.
"""

import ctypes
import json
import os
import sys


_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preload.json")

if sys.platform == "linux" and os.path.exists(_CONFIG):
    with open(_CONFIG) as _config_file:
        _config = json.load(_config_file)
    for _path in _config["preload_libs"]:
        ctypes.CDLL(_path, mode=ctypes.RTLD_GLOBAL)
    for _directory in _config["nvidia_driver_dirs"]:
        _libcuda = os.path.join(_directory, "libcuda.so.1")
        if os.path.exists(_libcuda):
            ctypes.CDLL(_libcuda, mode=ctypes.RTLD_GLOBAL)
            os.environ.setdefault("TRITON_LIBCUDA_PATH", _directory)
            break
