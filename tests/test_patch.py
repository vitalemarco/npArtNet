import unittest
import numpy as np
from npArtNet.patch import array_to_dmx_matrix, values_to_universe
from npArtNet.data_types import patch_dtype


class TestPatchModule(unittest.TestCase):
    def setUp(self):
        # Create a simple patch map for 2 float values to DMX channels
        # src 0 -> universe 0, address 1
        # src 1 -> universe 1, address 512
        # src 2 -> universe 0, address 256
        self.patch_map = np.array(
            [(0, 0, 1), (1, 1, 512), (2, 0, 256)], dtype=patch_dtype
        )

        # Test input floats
        self.source_array = np.array([1.0, 0.5, 0.0], dtype=np.float32)

    def test_array_to_dmx_matrix(self):
        unique_universes, matrix = array_to_dmx_matrix(
            self.source_array, self.patch_map
        )

        self.assertEqual(unique_universes, [0, 1])
        self.assertEqual(
            matrix.shape, (2, 512)
        )  # Maximum DMX Size (since we patched 512)

        # Verify 1.0 mapped to 255 at univ 0, channel 1 (index 0)
        self.assertEqual(matrix[0, 0], 255)

        # Verify 0.5 mapped to 127 at univ 1, channel 512 (index 511)
        self.assertEqual(matrix[1, 511], 127)

        # Verify 0.0 mapped to 0 at univ 0, channel 256 (index 255)
        self.assertEqual(matrix[0, 255], 0)

        # Verify empty channels are 0
        self.assertEqual(matrix[0, 10], 0)

    def test_values_to_universe(self):
        univ0_data = values_to_universe(
            self.source_array, self.patch_map, target_universe=0
        )
        univ1_data = values_to_universe(
            self.source_array, self.patch_map, target_universe=1
        )

        self.assertEqual(univ0_data.shape, (512,))
        self.assertEqual(univ1_data.shape, (512,))

        # Universe 0 has source[0] at idx 0 and source[2] at idx 255
        self.assertEqual(univ0_data[0], 255)
        self.assertEqual(univ0_data[255], 0)

        # Universe 1 has source[1] at idx 511
        self.assertEqual(univ1_data[511], 127)


if __name__ == "__main__":
    unittest.main()
