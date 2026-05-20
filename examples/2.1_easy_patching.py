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
