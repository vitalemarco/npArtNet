import os
import time
import numpy as np
from npArtNet import ArtnetServer


def main():
    """
    Test server tailored for 3.1_simple_engine.py
    Watches Universes 0 and 1 for Dimmer, RGB, and Moving Spot fixtures.
    """
    host = "127.0.0.1"
    if os.name == "nt":
        os.system("")

    print(f"Starting Simple Engine Server on {host}:6454...")
    print("Run `3.1_simple_engine.py` in another terminal to see incoming data!")
    print("Press Ctrl+C to stop.\n")

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
        except KeyboardInterrupt:
            pass

    print("\nShut down successfully.")


if __name__ == "__main__":
    main()
