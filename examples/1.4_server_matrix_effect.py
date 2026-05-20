import os
import time
import numpy as np
from npArtNet import ArtnetServer


def main():
    """
    Test server tailored for 3.2_matrix_effect.py
    Watches Universe 0 dynamically displaying up to 192 channels.
    """
    host = "127.0.0.1"
    if os.name == "nt":
        os.system("")

    print(f"Starting Matrix Effect Server on {host}:6454...")
    print("Run `3.2_matrix_effect.py` in another terminal to see incoming data!")
    print("Press Ctrl+C to stop.\n")

    with ArtnetServer(universes=[0], host=host) as server:
        np.set_printoptions(formatter={"int": lambda x: f"{x:3d}"}, linewidth=250)
        try:
            while server.is_running:
                state = server.get_matrix()
                print("\033[H\033[J", end="")
                print("--- 3.2 Matrix Effect (Universe 0) ---")
                print("Showing 192 channels (64 RGB Pixels):\n")

                # Reshape for nicer reading: 8 rows of 24 values (8 pixels w/ 3 channels each)
                # Ensure the slice matches 192 before reshaping
                data = state[0, 0:192]
                if len(data) == 192:
                    print(data.reshape((8, 24)))
                else:
                    # Fallback just in case
                    print(data)

                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    print("\nShut down successfully.")


if __name__ == "__main__":
    main()
