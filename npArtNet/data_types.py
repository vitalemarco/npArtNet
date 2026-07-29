"""
Defines the core structured arrays and constants for the npArtNet package.
"""

import numpy as np

DMX_UNIVERSE_SIZE = 512
"""int: The number of DMX channels in a single Art-Net universe."""

patch_dtype = np.dtype(
    [
        ("src", np.int32),
        ("universe", np.int16),
        ("address", np.int16),
    ]
)
"""
np.dtype
    A structured NumPy data type for defining pixel/fixture patches.
    Contains 'src' (int32 flattened array index), 'universe' (int16),
    and 'address' (int16, 1-based DMX address).
"""
