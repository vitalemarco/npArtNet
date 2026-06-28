import socket
import time
import unittest

import numpy as np

from npArtNet import ArtnetClient, ArtnetServer
from npArtNet.patch import array_to_dmx_matrix
from npArtNet.utils import get_msb_lsb
from npGdtf import GdtfError, Rig, load_gdtf_from_xml
from tests.gdtf_sample import SAMPLE_DESCRIPTION_XML


class TestGdtfPatch(unittest.TestCase):
    def setUp(self):
        self.fixture_type = load_gdtf_from_xml(SAMPLE_DESCRIPTION_XML)

    def test_patch_map_rows(self):
        rig = Rig()
        rig.patch(self.fixture_type, universe=0, address=10)
        patch_map = rig.build_patch_map()

        # Dimmer -> addr 10, Pan MSB -> 11, Pan LSB -> 12.
        rows = [tuple(int(v) for v in r) for r in patch_map]
        self.assertEqual(rows, [(0, 0, 10), (1, 0, 11), (2, 0, 12)])

    def test_multiple_fixtures_get_contiguous_src(self):
        rig = Rig()
        a = rig.patch(self.fixture_type, universe=0, address=1)
        b = rig.patch(self.fixture_type, universe=1, address=1)
        rig.build_patch_map()

        # First fixture takes src 0..2, second continues at 3..5.
        self.assertEqual(a.get_patch(), [(0, 0, 1), (1, 0, 2), (2, 0, 3)])
        self.assertEqual(b.get_patch(), [(3, 1, 1), (4, 1, 2), (5, 1, 3)])
        self.assertEqual(rig.total_slots, 6)

    def test_defaults_applied(self):
        rig = Rig()
        rig.patch(self.fixture_type, universe=0, address=1)
        rig.build_patch_map()
        state = rig.get_state()

        # Pan default 32768 (16-bit) -> MSB 128, LSB 0.
        msb, lsb = get_msb_lsb(32768)
        self.assertAlmostEqual(state[1], msb / 255.0, places=6)
        self.assertAlmostEqual(state[2], lsb / 255.0, places=6)

    def test_set_8bit_attribute(self):
        rig = Rig()
        fix = rig.patch(self.fixture_type, universe=0, address=10)
        patch_map = rig.build_patch_map()

        fix.set("Dimmer", 1.0)
        _, matrix = array_to_dmx_matrix(rig.get_state(), patch_map)
        # Address 10 -> index 9.
        self.assertEqual(matrix[0, 9], 255)

    def test_set_16bit_attribute_splits_correctly(self):
        rig = Rig()
        fix = rig.patch(self.fixture_type, universe=0, address=10)
        patch_map = rig.build_patch_map()

        fix.set("Pan", 0.5)
        expected_raw = round(0.5 * 65535)
        exp_msb, exp_lsb = get_msb_lsb(expected_raw)

        _, matrix = array_to_dmx_matrix(rig.get_state(), patch_map)
        # Pan MSB at address 11 (idx 10), LSB at address 12 (idx 11).
        self.assertEqual(matrix[0, 10], exp_msb)
        self.assertEqual(matrix[0, 11], exp_lsb)

    def test_set_raw_clamps(self):
        rig = Rig()
        fix = rig.patch(self.fixture_type, universe=0, address=1)
        patch_map = rig.build_patch_map()

        fix.set_raw("Pan", 999999)  # well above 65535
        _, matrix = array_to_dmx_matrix(rig.get_state(), patch_map)
        self.assertEqual(matrix[0, 1], 255)  # Pan MSB
        self.assertEqual(matrix[0, 2], 255)  # Pan LSB

    def test_unknown_attribute_raises(self):
        rig = Rig()
        fix = rig.patch(self.fixture_type, universe=0, address=1)
        rig.build_patch_map()
        with self.assertRaises(GdtfError):
            fix.set("NotAnAttribute", 1.0)

    def test_set_before_build_raises(self):
        rig = Rig()
        fix = rig.patch(self.fixture_type, universe=0, address=1)
        with self.assertRaises(GdtfError):
            fix.set("Dimmer", 1.0)

    def test_address_overflow_raises(self):
        rig = Rig()
        with self.assertRaises(GdtfError):
            # Footprint of 3 starting at 511 would overflow 512.
            rig.patch(self.fixture_type, universe=0, address=511)


class TestGdtfLoopback(unittest.TestCase):
    """End-to-end: GDTF-built frame over loopback, verified at the receiver."""

    def _free_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_gdtf_frame_round_trips(self):
        port = self._free_port()
        server = ArtnetServer(host="127.0.0.1", port=port, universes=[0])
        client = ArtnetClient(target_ip="127.0.0.1", port=port, universes=[0])
        server.start()
        try:
            ft = load_gdtf_from_xml(SAMPLE_DESCRIPTION_XML)
            rig = Rig()
            fix = rig.patch(ft, universe=0, address=1)
            client.set_patch(rig.build_patch_map())

            fix.set("Dimmer", 1.0)
            fix.set("Pan", 0.5)

            client.set_patched_dmx_values(rig.get_state())
            client.send_package()
            time.sleep(0.1)

            matrix = server.get_matrix()
            exp_msb, exp_lsb = get_msb_lsb(round(0.5 * 65535))
            self.assertEqual(matrix[0, 0], 255)      # Dimmer at addr 1
            self.assertEqual(matrix[0, 1], exp_msb)  # Pan MSB at addr 2
            self.assertEqual(matrix[0, 2], exp_lsb)  # Pan LSB at addr 3
        finally:
            server.close()
            client.close()


if __name__ == "__main__":
    unittest.main()
