import unittest
import numpy as np
from npArtNet.client import ArtnetClient


class TestArtnetClient(unittest.TestCase):
    def setUp(self):
        # We don't intend to actually send through the network in standard client tests
        # We can pass an invalid target just to verify stringing buffers together
        self.client = ArtnetClient(
            target_ip="127.0.0.1", universes=[0, 1, 2], packet_size=512
        )

    def tearDown(self):
        self.client.close()

    def test_initialization(self):
        self.assertEqual(self.client.num_universes, 3)
        self.assertEqual(self.client.buffer.shape, (3, 512))
        self.assertEqual(len(self.client.headers), 3)
        self.assertEqual(len(self.client.packets), 3)

    def test_add_universe(self):
        idx = self.client.add_universe(5)
        self.assertEqual(idx, 3)
        self.assertEqual(self.client.num_universes, 4)
        self.assertIn(5, self.client.universes)

    def test_set_dmx_value(self):
        # Test applying a raw array directly to a specific universe
        test_data = np.array([255, 128, 64, 32], dtype=np.uint8)
        self.client.set_dmx_value(universe=1, data=test_data)

        # Grab from the internal buffer
        internal_idx = self.client.universe_map[1]
        self.assertEqual(self.client.buffer[internal_idx, 0], 255)
        self.assertEqual(self.client.buffer[internal_idx, 1], 128)
        self.assertEqual(self.client.buffer[internal_idx, 5], 0)  # Defaults should be 0


if __name__ == "__main__":
    unittest.main()
