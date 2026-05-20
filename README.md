# npArtNet

A high-performance, NumPy-backed Art-Net matrix client, server, and patcher for Python.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Architecture Philosophy](#architecture-philosophy)
- [Examples](#examples)
    - [1. The "Easy Mode" (Client-Owned Patch)](#1-the-easy-mode-client-owned-patch)
    - [2. The "Pipeline Mode" (Stateless Math)](#2-the-pipeline-mode-stateless-math)
    - [3. Loopback Server](#3-loopback-server)

`npArtNet` utilizes a vectorized architecture for routing, sending, and receiving massive amounts of DMX data over the Art-Net protocol using NumPy arrays. It avoids high-level loops in Python and delegates the heavy lifting to C via NumPy, allowing it to easily handle standard 60-FPS continuous lighting loops across dozens of universes with minimal performance footprint.

## Features

- **Vectorized Patching:** Map flattened arrays of normalized floats directly to DMX universes and addresses almost instantly.
- **Dynamic Packet Sizing:** Automatically shrinks UDP payload sizes based on the highest patched address to save network bandwidth.
- **Zero-Copy Server:** An $O(1)$ routed receiver that maps incoming UDP packets directly into a 2D NumPy array structure (fast local loopback testing).
- **Engine Agnostic:** Built to accept generic float arrays (`0.0` to `1.0`), leaving 16-bit splits, GDTF logic, and fixture interpretation to your higher-level engine.

## Installation

As this is a custom local module, you can install it via pip straight from the directory:

```bash
pip install -e .
```

## Architecture Philosophy

`npArtNet` intentionally does **not** know what a "Moving Head" or a "16-bit Pan channel" is.

- **Your Engine (Smart Logic):** Maintains fixture state, handles timeline animations, generates 16-bit splits, and outputs a flat array of normalized floats (`0.0` to `1.0`).
- **npArtNet (Dumb Math):** Takes that flat float array, scales it to 8-bit `uint8`, and blasts it into the correct network packets as fast as physically possible.

## Examples

You can find complete runnable examples in the `examples/` directory.

### 1. The "Easy Mode" (Client-Owned Patch)

See `examples/01_easy_patching.py` for how to bind a rig layout to the client and constantly update array values.

```python
import numpy as np
import time
from npArtNet import ArtnetClient, patch_dtype


def main():
    # 1. Initialize the client
    # By default, port is 6454
    client = ArtnetClient(target_ip="127.0.0.1")

    # 2. Register your patch map (auto-expands universes and optimizes routing)
    # The columns are: (source_index, universe, dmx_address)
    # Note: DMX addresses should be standard 1-512. The client internally handles 0-based indexing.
    my_rig_patch = np.array(
        [
            (0, 0, 1),  # src index 0 -> Universe 0, Address 1 (e.g. Red)
            (1, 0, 2),  # src index 1 -> Universe 0, Address 2 (e.g. Green)
            (2, 0, 3),  # src index 2 -> Universe 0, Address 3 (e.g. Blue)
        ],
        dtype=patch_dtype,
    )

    client.set_patch(my_rig_patch)

    print("Sending Art-Net DMX data to localhost. Press Ctrl+C to stop.")

    # 3. Inside your Render Loop (e.g., 60 FPS)
    try:
        phase = 0.0
        while True:
            # Generate some simulated engine data (floats from 0.0 to 1.0)
            # Fading the values using a sine wave
            r = (np.sin(phase) + 1.0) / 2.0
            g = (np.sin(phase + 2.0) + 1.0) / 2.0
            b = (np.sin(phase + 4.0) + 1.0) / 2.0

            engine_state = np.array([r, g, b], dtype=np.float32)

            # The client scales, clips, and maps them directly into pre-allocated memory
            client.set_patched_dmx_values(engine_state)
            client.send_package()

            phase += 0.1
            time.sleep(1 / 60)  # 60 FPS

    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        print("Stopped client.")


if __name__ == "__main__":
    main()
```

### 2. The "Pipeline Mode" (Stateless Math)

See `examples/02_pipeline_matrices.py` for dealing with dynamic base arrays and overrides directly.

```python
import numpy as np
from npArtNet import ArtnetClient, patch_dtype
from npArtNet.patch import array_to_dmx_matrix


def main():
    """
    For power users who need to generate and blend multiple matrices
    (e.g., adding a strobe override over a pixel map) before
    sending them to the network.
    """
    client = ArtnetClient(target_ip="127.0.0.1")

    # Define some separate patches and values
    base_patch = np.array([(0, 0, 1), (1, 0, 2), (2, 0, 3)], dtype=patch_dtype)

    strobe_patch = np.array([(0, 0, 1), (1, 0, 2), (2, 0, 3)], dtype=patch_dtype)

    # Base values (dimmed to 50%)
    base_floats = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    # Strobe values (Full white, dominating the system)
    strobe_floats = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Generate two separate matrices dynamically
    # Matrix will automatically size itself based on max DMX address
    _, base_matrix = array_to_dmx_matrix(base_floats, base_patch)
    _, strobe_matrix = array_to_dmx_matrix(strobe_floats, strobe_patch)

    # Note: If the actual maximum universe/address between masks differed,
    # you'd pad the arrays to matching sizes first before blending.

    # HTP (Highest Takes Precedence)
    final_matrix = np.maximum(base_matrix, strobe_matrix)

    # Broadcast the final frame
    client.set_dmx_matrix(final_matrix)
    client.send_package()

    client.close()
    print("Final pipeline matrix sent to localhost.")
    print("Matrix data:")
    print(final_matrix)


if __name__ == "__main__":
    main()
```

### 3. Loopback Server

See `examples/03_local_server.py` for validating your outgoing transmissions.

```python
import time
import numpy as np
from npArtNet import ArtnetServer


def main():
    """
    Test your multi-universe math without plugging in a single piece
    of hardware using the zero-copy server.
    """
    # We will listen on the loopback interface on Universes 0 and 1
    host = "127.0.0.1"

    print(f"Starting Art-Net server on {host}:6454...")
    print("Run `01_easy_patching.py` in another terminal to see incoming data!")
    print("Press Ctrl+C to stop.\n")

    with ArtnetServer(universes=[0, 1], host=host) as server:

        # Configure numpy print options to keep terminal neat
        np.set_printoptions(formatter={"int": lambda x: f"{x:3d}"}, linewidth=100)

        try:
            while server.is_running:
                # Thread-safe snapshot of the current states
                current_lighting_state = server.get_matrix()

                # Let's inspect universe 0, channels 1->10 (matrix columns 0->9)
                uni0_ch1_10 = current_lighting_state[0, 0:10]

                # Print over the same line
                print(f"Universe 0 | CH 1-10: {uni0_ch1_10}", end="\r")

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass

    print("\nShut down successfully.")


if __name__ == "__main__":
    main()
```
