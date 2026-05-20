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
