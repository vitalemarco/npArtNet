from .data_types import DMX_UNIVERSE_SIZE
import numpy as np


def array_to_dmx_matrix(
    source_array: np.ndarray,
    patch_map: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """
    Map a generic array of normalized floats into a 2D DMX matrix.

    This function flattens the input array, scales normalized floats (0.0 to 1.0)
    to 8-bit DMX values (0 to 255), and maps them to their respective universes
    and addresses based on the patch map. Invalid user addresses are safely ignored.
    The resulting matrix is dynamically sized to the maximum required length.

    Parameters
    ----------
    source_array : np.ndarray
        The input values to map. Can be any shape, but must contain normalized floats.
    patch_map : np.ndarray
        A structured array (patch_dtype) containing 'src', 'universe', and 'address'.

    Returns
    -------
    tuple[list[int], np.ndarray]
        A tuple containing:
        - A list of the unique universes generated.
        - A 2D uint8 matrix of shape (num_universes, max_address) with the DMX data.
    """

    # FLATTEN ARRAY AND CLIP VALUES TO [0, 255]
    flat_array = (
        np.clip(np.multiply(source_array, 255.0), 0, 255).astype(np.uint8).ravel()
    )

    # CONVERT 1-BASED DMX-ADDRESS TO 0-BASED ARRAY INDICES
    raw_dest_address = patch_map["address"] - 1

    # MASK INVALID VALUES
    valid_mask = (raw_dest_address >= 0) & (raw_dest_address < DMX_UNIVERSE_SIZE)

    # PATCH MAPPING
    src = patch_map["src"]
    dest_universe = patch_map["universe"]
    dest_address = raw_dest_address[valid_mask]

    # IF THE PATCH MAP IS INVALID RETURN NO UNIVERSES AND A EMPTY MATRIX
    if len(dest_address) == 0:
        return [], np.zeros((0, 0), dtype=np.uint8)

    # EXTRACT UNIQUE UNIVERSES TO DEFINE THE MATRIX SIZE
    unique_universes = np.unique(dest_universe)
    num_rows = len(unique_universes)

    # DYNAMIC MAX WIDTH OF THE MATRIX
    max_index = np.max(dest_address)
    max_len = min(int(max_index) + 1, DMX_UNIVERSE_SIZE)

    # BUILD THE MATRIX OF UNIVERSES / VALUES
    matrix = np.zeros((num_rows, max_len), dtype=np.uint8)

    # CREATE A SORTED ARRAY OF UNIVERSE INDICES
    row_indices = np.searchsorted(unique_universes, dest_universe)

    # MAP INPUT VALUES TO THE UNIVERSE/ADDRESS MATRIX
    matrix[row_indices, dest_address] = flat_array[src]

    return unique_universes.tolist(), matrix


def values_to_universe(
    source_values: np.ndarray, patch_map: np.ndarray, target_universe: int
) -> np.ndarray:
    """
    Extract and map data exclusively for a single specified universe.

    Ideal for lightweight single-universe setups or feeding `set_dmx_value()`.

    Parameters
    ----------
    source_values : np.ndarray
        The input values to map (normalized floats 0.0 to 1.0).
    patch_map : np.ndarray
        A structured array (patch_dtype) containing 'src', 'universe', and 'address'.
    target_universe : int
        The specific universe number to extract data for.

    Returns
    -------
    np.ndarray
        A 1D uint8 array of length 512 containing the DMX data for the target universe.
    """

    # FLATTEN ARRAY AND CLIP VALUES TO [0, 255]
    flat_values = (
        np.clip(np.multiply(source_values, 255.0), 0, 255).astype(np.uint8).ravel()
    )

    # CREATE MASK FOR TARGET UNIVERSE
    universe_mask = patch_map["universe"] == target_universe

    # FILTER THE PATCH MAP
    filtered_src = patch_map["src"][universe_mask]
    filtered_address = patch_map["address"][universe_mask] - 1

    # CREATE ONE UNIVERSE DATA
    universe_data = np.zeros(512, dtype=np.uint8)
    universe_data[filtered_address] = flat_values[filtered_src]

    return universe_data
