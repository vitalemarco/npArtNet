import numpy as np
import time
from npArtNet import ArtnetClient, patch_dtype


def build_canvas_patch(width: int, height: int, start_universe: int) -> np.ndarray:
    """
    Build a patch for one RGB canvas with local src indices.

    Each pixel consumes 3 consecutive DMX addresses (R, G, B). The 'src'
    indices stay local to this canvas (0 .. width*height*3 - 1); the client
    shifts them into the concatenated frame space inside set_patches().
    """
    num_channels = width * height * 3
    src = np.arange(num_channels, dtype=np.int32)
    universe = start_universe + (src // 512).astype(np.int16)
    address = (src % 512 + 1).astype(np.int16)

    patch = np.zeros(num_channels, dtype=patch_dtype)
    patch["src"] = src
    patch["universe"] = universe
    patch["address"] = address
    return patch


def main():
    # 1. Initialize the client
    client = ArtnetClient(target_ip="127.0.0.1")

    # 2. Two independent 16x16 RGB canvases (256 pixels = 768 floats each),
    # wired to different universe ranges of the same Art-Net node
    width, height = 16, 16
    frame_length = width * height * 3  # 768

    zone_a_patch = build_canvas_patch(width, height, start_universe=0)
    zone_b_patch = build_canvas_patch(width, height, start_universe=5)

    # 3. Bind both patches ONCE, at configuration time.
    # frame_lengths must be passed in: a canvas can be larger than its
    # wired patch, so lengths cannot be derived from the patch maps.
    client.set_patches([zone_a_patch, zone_b_patch], [frame_length, frame_length])

    print("Sending two canvases to localhost. Press Ctrl+C to stop.")

    # 4. Inside your Render Loop (e.g., 60 FPS)
    try:
        phase = 0.0
        while True:
            # Each zone renders its own independent frame (floats 0.0 to 1.0)
            x = np.linspace(0, 1, width, dtype=np.float32)
            y = np.linspace(0, 1, height, dtype=np.float32)[:, None]

            wave_a = (np.sin(phase + x * 4.0) + 1.0) / 2.0
            zone_a_frame = np.broadcast_to(wave_a, (height, width))
            zone_a_frame = np.repeat(zone_a_frame[..., None], 3, axis=2)

            wave_b = (np.cos(phase + y * 4.0) + 1.0) / 2.0
            zone_b_frame = np.broadcast_to(wave_b, (height, width))
            zone_b_frame = np.repeat(zone_b_frame[..., None], 3, axis=2)

            # Multi-source contract: concatenate frames in patch order,
            # one indexed write routes everything, one burst sends it all.
            frame = np.concatenate([zone_a_frame.ravel(), zone_b_frame.ravel()]).astype(
                np.float32
            )

            client.set_patched_dmx_values(frame)
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
