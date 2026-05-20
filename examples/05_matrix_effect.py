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
                    frame_data = np.random.randint(0, 2, total_channels).astype(
                        np.float32
                    )
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
        print("\nMatrix effect stopped.")


if __name__ == "__main__":
    main()
