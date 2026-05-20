import os
import subprocess
import time
import numpy as np
from npArtNet import ArtnetServer


def main():
    """
    Test server tailored for 2.2_pipeline_matrices.py
    Watches Universe 0 for the HTP blended matrix output.
    """
    host = "127.0.0.1"
    if os.name == "nt":
        subprocess.run("", shell=True)

    print(f"Starting Pipeline Matrices Server on {host}:6454...")
    print("Run `2.2_pipeline_matrices.py` in another terminal to see incoming data!")
    print("Press Ctrl+C to stop.\n")

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
        except KeyboardInterrupt:
            pass

    print("\nShut down successfully.")


if __name__ == "__main__":
    main()
