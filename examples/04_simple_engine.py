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
        self.src_start = 0  # Will be assigned by the engine memory allocator

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
        return [(self.src_start, self.universe, self.dmx_start)]

    def update(self, state_array: np.ndarray, time_sec: float):
        # Pulsing effect based on time
        state_array[self.src_start] = (math.sin(time_sec * 3.0) + 1.0) / 2.0


class RGBFixture(Fixture):
    """A generic 3-channel RGB light."""

    num_channels = 3

    def get_patch(self):
        return [
            (self.src_start + 0, self.universe, self.dmx_start + 0),  # Red
            (self.src_start + 1, self.universe, self.dmx_start + 1),  # Green
            (self.src_start + 2, self.universe, self.dmx_start + 2),  # Blue
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
            (
                self.src_start + 0,
                self.universe,
                self.dmx_start + 0,
            ),  # Pan (0-1 represents 0-540deg)
            (
                self.src_start + 1,
                self.universe,
                self.dmx_start + 1,
            ),  # Tilt (0-1 represents 0-270deg)
            (self.src_start + 2, self.universe, self.dmx_start + 2),  # Dimmer
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

    print(
        f"Engine compiled with {engine.total_channels} active channels across {len(client.universes)} universe(s)."
    )
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

            time.sleep(1 / 60)  # 60 FPS limiter

    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        print("\nEngine stopped.")


if __name__ == "__main__":
    main()
