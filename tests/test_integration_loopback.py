import unittest
import time
import numpy as np
import socket
from npArtNet.client import ArtnetClient
from npArtNet.server import ArtnetServer


class TestIntegrationLoopback(unittest.TestCase):
    def get_free_port(self):
        """Helper to get a random free port so we don't conflict with other services."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def setUp(self):
        # We use a random port to ensure clean UDP bindings during CI/tests
        self.port = self.get_free_port()
        self.universes = [3, 7]

        self.server = ArtnetServer(
            host="127.0.0.1", port=self.port, universes=self.universes
        )
        self.client = ArtnetClient(
            target_ip="127.0.0.1", port=self.port, universes=self.universes
        )
        self.server.start()

    def tearDown(self):
        self.server.close()
        self.client.close()

    def test_loopback_transmission(self):
        # 1. Update Client Buffer for universe 3 and universe 7
        data_u3 = np.array([11, 22, 33, 44], dtype=np.uint8)
        data_u7 = np.array([255, 127, 0, 64], dtype=np.uint8)

        self.client.set_dmx_value(universe=3, data=data_u3)
        self.client.set_dmx_value(universe=7, data=data_u7)

        # 2. Transmit through local loopback network adapter
        self.client.send_package()

        # Give the server thread a tiny fraction of a second to ingest the UDP packets
        time.sleep(0.1)

        # 3. Request matrix from ArtNet server daemon and verify
        matrix = self.server.get_matrix()

        # Verify Universe 3 (Row 0)
        self.assertEqual(matrix[0, 0], 11)
        self.assertEqual(matrix[0, 1], 22)
        self.assertEqual(matrix[0, 3], 44)

        # Verify Universe 7 (Row 1)
        self.assertEqual(matrix[1, 0], 255)
        self.assertEqual(matrix[1, 1], 127)

        # Verify untoched channels remain zero
        self.assertEqual(matrix[0, 100], 0)


if __name__ == "__main__":
    unittest.main()
