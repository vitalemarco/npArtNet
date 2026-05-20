"""
npArtNet
=====

A high-performance, NumPy-backed Art-Net matrix client, server, and patcher.
This package provides a vectorized architecture for routing, sending, and receiving
massive amounts of DMX data over the Art-Net protocol using NumPy arrays.

Features
-----
- **Vectorized Patching:** Map flattened arrays of normalized floats directly to DMX universes and addresses instantly.
- **Dynamic Packet Sizing:** Automatically shrinks UDP payload sizes based on the highest patched address to save network bandwidth.
- **Zero-Copy Server:** An O(1) routed receiver that maps incoming UDP packets directly into a 2D NumPy array for high-speed local loopback testing.
- **Engine Agnostic:** Built to accept generic float arrays (`0.0` to `1.0`), leaving 16-bit splits and fixture logic to your higher-level engine.

Examples
-----

### 1. The "Easy Mode" (Client-Owned Patch)

If you want the client to manage the routing for you, simply register a NumPy structured array (`patch_dtype`) containing your `src`, `universe`, and `address` mapping.

```python
import numpy as np
from npArtNet import ArtnetClient, patch_dtype

# 1. Initialize the client
client = ArtnetClient(target_ip="10.0.0.5")

# 2. Register your patch map (auto-expands universes and optimizes routing)
# Note: DMX addresses should be standard 1-512. The client handles 0-based conversions safely.
my_rig_patch = np.array([
    (0, 0, 1), # src index 0 -> Universe 0, Address 1 (Red)
    (1, 0, 2), # src index 1 -> Universe 0, Address 2 (Green)
    (2, 0, 3), # src index 2 -> Universe 0, Address 3 (Blue)
], dtype=patch_dtype)

client.set_patch(my_rig_patch)

# 3. Inside your Render Loop (60 FPS)
# Pass an array of floats (0.0 - 1.0). The client scales, clips, and maps them instantly.
engine_state = np.array([1.0, 0.5, 0.0], dtype=np.float32)

client.set_patched_dmx_values(engine_state)
client.send_package()
```

### 2. The "Pipeline Mode" (Stateless Math)
For power users who need to generate and blend multiple matrices (e.g., adding a strobe override over a pixel map) before sending them to the network.

```python
import numpy as np
from npArtNet import ArtnetClient
from npArtNet.patch import array_to_dmx_matrix

client = ArtnetClient(target_ip="10.0.0.5")

# Generate two separate matrices dynamically
_, base_matrix = array_to_dmx_matrix(base_floats, base_patch)
_, strobe_matrix = array_to_dmx_matrix(strobe_floats, strobe_patch)

# Blend them using standard NumPy operations
final_matrix = np.maximum(base_matrix, strobe_matrix)

# Broadcast the final frame
client.set_dmx_matrix(final_matrix)
client.send_package()

```

### 3. Local Loopback Testing
Test your multi-universe math without plugging in a single piece of hardware using the zero-copy server.

```python
from npArtNet import ArtnetServer

# Listen to Universes 0 and 1 on localhost
with ArtnetServer(universes=[0, 1], host="127.0.0.1") as server:

    # ... send some data with ArtnetClient to 127.0.0.1 ...

    # Instantly grab the 2D matrix of current network data
    current_lighting_state = server.get_matrix()
    print(current_lighting_state)

```

Architecture Philosophy: Smart Logic vs. Dumb Math
-----

`npArtNet` intentionally does **not** know what a "Moving Head" or a "16-bit Pan channel" is.

- **Your Engine (Smart Logic):** Loads GDTF files, tracks fixture state, processes 16-bit math, and outputs a flat array of normalized floats (`0.0` to `1.0`).
- **npArtNet (Dumb Math):** Takes that flat array and blasts it into the correct network packets as fast as physically possible.
"""

from .data_types import patch_dtype
from .utils import clamp_value, get_msb_lsb, make_address_mask
from .patch import array_to_dmx_matrix, values_to_universe
from .client import ArtnetClient
from .server import ArtnetServer
