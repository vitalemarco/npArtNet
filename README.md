# npArtNet

A high-performance, NumPy-backed Art-Net matrix client, server, and patcher for Python.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Architecture Philosophy](#architecture-philosophy)
- [Examples](#examples)
    - [1. Server Tools](#1-server-tools)
        - [1.1 Server for Easy Patching](#11-server-for-easy-patching)
        - [1.2 Server for Pipeline Matrices](#12-server-for-pipeline-matrices)
        - [1.3 Server for Simple Engine](#13-server-for-simple-engine)
        - [1.4 Server for Matrix Effect](#14-server-for-matrix-effect)
    - [2. Simple Raw Value Sends](#2-simple-raw-value-sends)
        - [2.1 The "Easy Mode" (Client-Owned Patch)](#21-the-easy-mode-client-owned-patch)
        - [2.2 The "Pipeline Mode" (Stateless Math)](#22-the-pipeline-mode-stateless-math)
    - [3. Engine Driven](#3-engine-driven)
        - [3.1 Simple Engine Architecture](#31-simple-engine-architecture)
        - [3.2 8x8 Matrix Random Effects](#32-8x8-matrix-random-effects)

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

### 1. Server Tools

Testing complex matrix applications without physical DMX hardware can be tricky. `npArtNet` features an $O(1)$-routed local Server for immediate feedback.

To easily test each client approach, we have created 4 specialized companion servers. They all bind to `127.0.0.1` in a background daemon thread to instantly parse and cleanly display incoming Art-Net broadcasts.

#### 1.1 Server for Easy Patching

Tailored for `2.1_easy_patching.py`. Watches Universe 0 and slices just the first 3 channels representing the RGB fading sine waves.

See `examples/1.1_server_easy_patching.py`.

```python
import os
import subprocess
import time
import numpy as np
from npArtNet import ArtnetServer

def main():
    host = "127.0.0.1"
    if os.name == "nt": subprocess.run("", shell=True)

    print(f"Starting Easy Patching Server on {host}:6454...")
    print("Run `2.1_easy_patching.py` in another terminal to see incoming data!\n")

    with ArtnetServer(universes=[0], host=host) as server:
        np.set_printoptions(formatter={"int": lambda x: f"{x:3d}"}, linewidth=100)
        try:
            while server.is_running:
                state = server.get_matrix()
                print("\033[H\033[J", end="")
                print("--- 2.1 Easy Patching (Universe 0) ---")
                print("Ch 1 (R) | Ch 2 (G) | Ch 3 (B)")
                print(state[0, 0:3])
                time.sleep(0.1)
        except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
```

#### 1.2 Server for Pipeline Matrices

Tailored for `2.2_pipeline_matrices.py`. Validates your mathematical HTP blend overrides.

See `examples/1.2_server_pipeline_matrices.py`.

```python
import os
import subprocess
import time
import numpy as np
from npArtNet import ArtnetServer

def main():
    host = "127.0.0.1"
    if os.name == "nt": subprocess.run("", shell=True)

    print(f"Starting Pipeline Matrices Server on {host}:6454...")
    print("Run `2.2_pipeline_matrices.py` in another terminal to see incoming data!\n")

    with ArtnetServer(universes=[0], host=host) as server:
        np.set_printoptions(formatter={"int": lambda x: f"{x:3d}"}, linewidth=100)
        try:
            while server.is_running:
                state = server.get_matrix()
                print("\033[H\033[J", end="")
                print("--- 2.2 Pipeline Matrices (Universe 0) ---")
                print("Ch 1-3 (Blended Base + Strobe)")
                print(state[0, 0:3])
                time.sleep(0.1)
        except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
```

#### 1.3 Server for Simple Engine

Tailored for `3.1_simple_engine.py`. Demonstrates safely listening to multiple universes simultaneously.

See `examples/1.3_server_simple_engine.py`.

```python
import os
import subprocess
import time
import numpy as np
from npArtNet import ArtnetServer

def main():
    host = "127.0.0.1"
    if os.name == "nt": subprocess.run("", shell=True)

    print(f"Starting Simple Engine Server on {host}:6454...")
    print("Run `3.1_simple_engine.py` in another terminal to see incoming data!\n")

    with ArtnetServer(universes=[0, 1], host=host) as server:
        np.set_printoptions(formatter={"int": lambda x: f"{x:3d}"}, linewidth=100)
        try:
            while server.is_running:
                state = server.get_matrix()
                print("\033[H\033[J", end="")
                print("--- 3.1 Simple Engine ---")
                print("Universe 0 [Ch 1 (Dim) | Ch 2-4 (RGB)]    :", state[0, 0:4])
                print("Universe 1 [Ch 1-2 (Pan/Tilt) | Ch 3 (Dim)]:", state[1, 0:3])
                time.sleep(0.1)
        except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
```

#### 1.4 Server for Matrix Effect

Tailored for `3.2_matrix_effect.py`. Shows the capability of parsing huge amounts of data instantly by reshaping 192 continuous channels into an 8x8 structural readout map.

See `examples/1.4_server_matrix_effect.py`.

```python
import os
import subprocess
import time
import numpy as np
from npArtNet import ArtnetServer

def main():
    host = "127.0.0.1"
    if os.name == "nt": subprocess.run("", shell=True)

    print(f"Starting Matrix Effect Server on {host}:6454...")
    print("Run `3.2_matrix_effect.py` in another terminal to see incoming data!\n")

    with ArtnetServer(universes=[0], host=host) as server:
        np.set_printoptions(formatter={"int": lambda x: f"{x:3d}"}, linewidth=250)
        try:
            while server.is_running:
                state = server.get_matrix()
                print("\033[H\033[J", end="")
                print("--- 3.2 Matrix Effect (Universe 0) ---")
                print("Showing 192 channels (64 RGB Pixels):\n")

                data = state[0, 0:192]
                if len(data) == 192:
                    print(data.reshape((8, 24)))
                else:
                    print(data)

                time.sleep(0.1)
        except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
```

### 2. Simple Raw Value Sends

#### 2.1 The "Easy Mode" (Client-Owned Patch)

This example is the recommended approach for most users. It demonstrates how to initialize the `ArtnetClient` and register a `patch_map` using the custom `patch_dtype`. The `patch_map` acts as the definitive roadmap for the client, detailing exactly how the 1D float array you generate in your logic translates to specific universes and DMX addresses.

**What it does:**

- It creates a tiny patch for three individual channels (Red, Green, Blue) at the beginning of Universe 0.
- It enters a continuous continuous 60 FPS while-loop, generating shifting values via a mathematical sine wave mapped to an array.
- Calling `client.set_patched_dmx_values(engine_state)` instantly maps the floats, scales them up to standard 0-255 DMX values, and organizes them perfectly into the pre-allocated internal network packets before dispatch.

See `examples/2.1_easy_patching.py` for how to bind a rig layout to the client and constantly update array values.

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

#### 2.2 The "Pipeline Mode" (Stateless Math)

If you are dealing with distinct layers of lighting states—such as a base sequence topped with a high-priority strobe effect mask—you can bypass client patching entirely and use `npArtNet` to evaluate states statelesssly.

**What it does:**

- Shows how to use the helper function `array_to_dmx_matrix` to generate fully constructed 2D representations of multiple different rig patches.
- Takes a dimmed `base_matrix` and a full-white `strobe_matrix`.
- Employs NumPy's `np.maximum()` function to execute a mathematically pristine "Highest Takes Precedence" (HTP) blend across all universes at once before injecting the final compiled matrix directly into the sender client via `client.set_dmx_matrix()`.

See `examples/2.2_pipeline_matrices.py` for dealing with dynamic base arrays and overrides directly.

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

### 3. Engine Driven

#### 3.1 Simple Engine Architecture

This is a comprehensive demonstration of how a complete "Lighting Engine" pairs perfectly with the philosophy of `npArtNet`.

**What it does:**

- Defines fixture object classes (Dimmer, RGB, Moving Spot) representing virtual models of physical lights, isolating their control logic.
- Utilizes `SimpleEngine` to calculate how many total array indices are required, safely linking the structural layout (Patch) to the specific math instructions embedded in fixtures.
- Centralizes runtime in `engine.tick(t)`, which calls `f.update()` upon every instantiated fixture so they can manipulate their specific region of the overarching `state` master-float-array.
- That one continuous array is then fired off into `npArtNet` which handles the complex process of turning a flat list of 0-1 values into standard DMX protocol distributions.

See `examples/3.1_simple_engine.py` for a more developed abstraction where a "Lighting Engine" manages different fixture types (Dimmers, RGB, Moving Spots). The engine tracks their memory allocations and outputs a single 1D flat array mapping perfectly to the `npArtNet` client.

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

#### 3.2 8x8 Matrix Random Effects

The true power of mapping floats directly to DMX using NumPy comes from exploiting standard algebraic and statistical arrays provided out-of-the-box by the library.

**What it does:**

- Programmatically constructs a continuous 8x8 RGB Pixel Matrix representing 192 successive DMX channel assignments.
- Uses `np.random` functions to generate complex behaviors matching the matrix scale instantly avoiding Python `for` loops entirely.
- Features a timer loop replacing the state completely every 1.0 seconds swapping between smooth float generation (static), boolean discrete colors, or masked logical thresholds targeting extreme sub-selections creating glitch-sparkles.

See `examples/3.2_matrix_effect.py` for a demonstration of programmatically generating an RGB matrix patch and manipulating it using fast `np.random` NumPy operations every second to create lighting states.

```python
import time
import numpy as np
from npArtNet import ArtnetClient, patch_dtype


def main():
    client = ArtnetClient(target_ip="127.0.0.1")

    # Create an 8x8 RGB Matrix Patch (64 pixels, 192 channels)
    # Universe 0, starting at DMX address 1
    num_pixels = 8 * 8
    channels_per_pixel = 3
    total_channels = num_pixels * channels_per_pixel

    # We can programmatically generate the patch list
    patch_list = []
    for i in range(total_channels):
        # src_index, universe, dmx_address (1-based)
        patch_list.append((i, 0, i + 1))

    matrix_patch = np.array(patch_list, dtype=patch_dtype)
    client.set_patch(matrix_patch)

    print(f"8x8 RGB Matrix patched: {total_channels} channels on Universe 0.")
    print("Running random effects... Press Ctrl+C to stop.")

    state = -1
    last_change_time = 0
    # Pre-allocate the float array
    frame_data = np.zeros(total_channels, dtype=np.float32)

    try:
        while True:
            current_time = time.time()

            # Change state and matrix values every 1 second
            if current_time - last_change_time >= 1.0:
                state = (state + 1) % 3

                if state == 0:
                    # State 0: Soft random RGB noise (0.0 to 1.0)
                    frame_data = np.random.rand(total_channels).astype(np.float32)
                    print("Effect State 0: Soft Random Noise     ", end="\r")

                elif state == 1:
                    # State 1: Hard digital colors (binary 0.0 or 1.0)
                    frame_data = np.random.randint(0, 2, total_channels).astype(np.float32)
                    print("Effect State 1: Hard Digital Colors   ", end="\r")

                elif state == 2:
                    # State 2: Black matrix with random glitch sparkles (mostly 0.0, some 1.0)
                    mask = np.random.rand(total_channels) > 0.95
                    frame_data.fill(0.0)
                    frame_data[mask] = 1.0
                    print("Effect State 2: Glitch Sparkles       ", end="\r")

                last_change_time = current_time

            # Route, scale, and dispatch the DMX frame
            client.set_patched_dmx_values(frame_data)
            client.send_package()

            # Run the network loop at 60 FPS
            time.sleep(1 / 60)

    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        print("\\nMatrix effect stopped.")


if __name__ == "__main__":
    main()
```
