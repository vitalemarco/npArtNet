import unittest
import numpy as np
from npArtNet.server import ArtnetServer
from npArtNet.utils import make_address_mask


class TestArtnetServer(unittest.TestCase):
    def setUp(self):
        self.server = ArtnetServer(universes=[0, 10, 20], host="127.0.0.1", port=6454)

    def tearDown(self):
        if self.server.is_running:
            self.server.close()

    def test_server_initialization(self):
        self.assertEqual(self.server.num_rows, 3)
        self.assertEqual(self.server.buffer.shape, (3, 512))

        # Check O(1) mask dictionary
        mask_0 = bytes(make_address_mask(0, 0, 0, True))
        mask_10 = bytes(make_address_mask(10, 0, 0, True))

        self.assertIn(mask_0, self.server.mask_to_row)
        self.assertIn(mask_10, self.server.mask_to_row)
        self.assertEqual(self.server.mask_to_row[mask_10], 1)

    def test_get_matrix_zero_state(self):
        matrix = self.server.get_matrix()
        self.assertEqual(matrix.shape, (3, 512))
        self.assertTrue(np.all(matrix == 0))


if __name__ == "__main__":
    unittest.main()
