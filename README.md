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
    - [4. Simple Engine Architecture](#4-simple-engine-architecture)

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

### 4. Simple Engine Architecture

See `examples/04_simple_engine.py` for a more developed abstraction where a "Lighting Engine" manages different fixture types (Dimmers, RGB, Moving Spots). The engine tracks their memory allocations and outputs a single 1D flat array mapping perfectly to the `npArtNet` client.

```python
import time
import math
import numpy as np
from npArtNet import ArtnetClient, patch_dtype


# ---------------------------------------------------------
# 1. FIXTURE DEFINITIONS (Smart Logic)
# ---------------------------------------------------------
class Fixture:
    """Base class for any fixture in our engine."""
    num_channels = 0

    def __init__(self, universe: int, dmx_start: int):
        self.universe = universe
        self.dmx_start = dmx_start
        self.src_start = 0 # Will be assigned by the engine memory allocator

    def get_patch(self) -> list[tuple[int, int, int]]:
        """Return the patch mappings for this fixture."""
        return []

    def update(self, state_array: np.ndarray, time_sec: float):
        """Update the fixture's designated floats in the master state array."""
        pass


class Dimmer(Fixture):
    """A generic 1-channel dimmer."""
    num_channels = 1

    def get_patch(self):
        return [
            (self.src_start, self.universe, self.dmx_start)
        ]

    def update(self, state_array: np.ndarray, time_sec: float):
        # Pulsing effect based on time
        state_array[self.src_start] = (math.sin(time_sec * 3.0) + 1.0) / 2.0


class RGBFixture(Fixture):
    """A generic 3-channel RGB light."""
    num_channels = 3

    def get_patch(self):
        return [
            (self.src_start + 0, self.universe, self.dmx_start + 0), # Red
            (self.src_start + 1, self.universe, self.dmx_start + 1), # Green
            (self.src_start + 2, self.universe, self.dmx_start + 2), # Blue
        ]

    def update(self, state_array: np.ndarray, time_sec: float):
        # RGB slow color cycle
        state_array[self.src_start + 0] = (math.sin(time_sec * 1.0) + 1.0) / 2.0
        state_array[self.src_start + 1] = (math.sin(time_sec * 1.5 + 2.0) + 1.0) / 2.0
        state_array[self.src_start + 2] = (math.sin(time_sec * 0.8 + 4.0) + 1.0) / 2.0


class MovingSpot(Fixture):
    """A 3-channel abstract moving spot (Pan, Tilt, Dimmer)."""
    num_channels = 3

    def get_patch(self):
        return [
            (self.src_start + 0, self.universe, self.dmx_start + 0), # Pan (0-1 represents 0-540deg)
            (self.src_start + 1, self.universe, self.dmx_start + 1), # Tilt (0-1 represents 0-270deg)
            (self.src_start + 2, self.universe, self.dmx_start + 2), # Dimmer
        ]

    def update(self, state_array: np.ndarray, time_sec: float):
        # Move in circles
        state_array[self.src_start + 0] = (math.sin(time_sec * 2.0) + 1.0) / 2.0
        state_array[self.src_start + 1] = (math.cos(time_sec * 2.0) + 1.0) / 2.0
        # Dimmer stays at full (1.0)
        state_array[self.src_start + 2] = 1.0


# ---------------------------------------------------------
# 2. ENGINE DEFINITION (Memory & Tick Management)
# ---------------------------------------------------------
class SimpleEngine:
    """Manages memory allocation for fixtures and builds the master patch."""
    def __init__(self):
        self.fixtures: list[Fixture] = []
        self.total_channels = 0
        self.state: np.ndarray | None = None

    def add_fixture(self, fixture: Fixture):
        # Assign a slice of the global float array to this fixture
        fixture.src_start = self.total_channels
        self.total_channels += fixture.num_channels
        self.fixtures.append(fixture)

    def build_patch_map(self) -> np.ndarray:
        # Pre-allocate the master state array based on total channels registered
        self.state = np.zeros(self.total_channels, dtype=np.float32)

        patch_list = []
        for f in self.fixtures:
            patch_list.extend(f.get_patch())

        return np.array(patch_list, dtype=patch_dtype)

    def tick(self, time_sec: float) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Engine patch map not built yet. Call build_patch_map() first.")

        # Ask each fixture to do its math and write to its assigned slice of `self.state`
        for f in self.fixtures:
            f.update(self.state, time_sec)

        return self.state


# ---------------------------------------------------------
# 3. MAIN LOOP (Connecting Engine to npArtNet)
# ---------------------------------------------------------
def main():
    engine = SimpleEngine()

    # Layout a tiny stage
    # Dimmer 1 on Universe 0, Address 1
    engine.add_fixture(Dimmer(universe=0, dmx_start=1))

    # RGB Fixture 1 on Universe 0, Address 2, 3, 4
    engine.add_fixture(RGBFixture(universe=0, dmx_start=2))

    # Moving Spot on Universe 1, Address 1, 2, 3
    engine.add_fixture(MovingSpot(universe=1, dmx_start=1))

    # Initialize Client and compile the rig layout
    client = ArtnetClient(target_ip="127.0.0.1")
    master_patch = engine.build_patch_map()
    client.set_patch(master_patch)

    print(f"Engine compiled with {engine.total_channels} active channels across {len(client.universes)} universe(s).")
    print("Transmitting. Press Ctrl+C to stop.")

    start_time = time.time()

    try:
        while True:
            t = time.time() - start_time

            # --- THE MAGIC HAPPENS HERE ---
            # 1. High-level engine calculates positions/colors as plain floats
            frame_floats = engine.tick(t)

            # 2. npArtNet instantly routes, scales (to 0-255), and dispatches it
            client.set_patched_dmx_values(frame_floats)
            client.send_package()

            time.sleep(1 / 60) # 60 FPS limiter

    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        print("\\nEngine stopped.")

if __name__ == "__main__":
    main()
```
