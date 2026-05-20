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
