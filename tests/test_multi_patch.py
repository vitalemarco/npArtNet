import unittest
import numpy as np
from npArtNet.client import ArtnetClient
from npArtNet.data_types import patch_dtype


class TestMultiPatch(unittest.TestCase):
    def setUp(self):
        self.client = ArtnetClient(
            target_ip="127.0.0.1", universes=[0], packet_size=512
        )

    def tearDown(self):
        self.client.close()

    def test_offsets_route_each_source(self):
        # Two sources, 4 floats each. Source 1's src indices are local (0-3)
        # and must be shifted into the concatenated frame space.
        patch_a = np.array(
            [(0, 0, 1), (1, 0, 2), (3, 0, 3)], dtype=patch_dtype
        )
        patch_b = np.array(
            [(0, 1, 1), (2, 1, 5)], dtype=patch_dtype
        )
        self.client.set_patches([patch_a, patch_b], [4, 4])

        frame_a = np.array([1.0, 0.5, 0.25, 0.75], dtype=np.float32)
        frame_b = np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float32)
        self.client.set_patched_dmx_values(np.concatenate([frame_a, frame_b]))

        row0 = self.client.universe_map[0]
        row1 = self.client.universe_map[1]

        self.assertEqual(self.client.buffer[row0, 0], 255)  # a[0]
        self.assertEqual(self.client.buffer[row0, 1], 127)  # a[1]
        self.assertEqual(self.client.buffer[row0, 2], 191)  # a[3]
        self.assertEqual(self.client.buffer[row1, 0], 0)    # b[0]
        self.assertEqual(self.client.buffer[row1, 4], 127)  # b[2]

    def test_frame_larger_than_patch_reserves_space(self):
        # Source A's canvas is larger than its wired patch: its declared frame
        # length (not max(src)+1) must determine source B's offset.
        patch_a = np.array([(0, 0, 1)], dtype=patch_dtype)  # only src 0 wired
        patch_b = np.array([(0, 0, 2)], dtype=patch_dtype)
        self.client.set_patches([patch_a, patch_b], [100, 10])

        frame = np.zeros(110, dtype=np.float32)
        frame[0] = 1.0    # source A, src 0
        frame[100] = 0.5  # source B, src 0 lives at index 100 in the concat
        self.client.set_patched_dmx_values(frame)

        row = self.client.universe_map[0]
        self.assertEqual(self.client.buffer[row, 0], 255)
        self.assertEqual(self.client.buffer[row, 1], 127)

    def test_single_patch_frame_lengths_optional(self):
        patch = np.array([(0, 0, 1), (1, 0, 2)], dtype=patch_dtype)
        self.client.set_patches([patch])  # no frame_lengths: allowed for one patch
        self.client.set_patched_dmx_values(np.array([1.0, 0.5]))
        row = self.client.universe_map[0]
        self.assertEqual(self.client.buffer[row, 0], 255)
        self.assertEqual(self.client.buffer[row, 1], 127)

    def test_set_patch_delegates(self):
        patch = np.array([(0, 0, 1), (5, 1, 512), (2, 0, 256)], dtype=patch_dtype)
        self.client.set_patch(patch)
        self.assertTrue(self.client.has_patch)
        np.testing.assert_array_equal(self.client._patch_src, [0, 5, 2])
        np.testing.assert_array_equal(self.client._patch_addr, [0, 511, 255])
        self.assertIn(1, self.client.universes)  # auto-registered

    def test_collision_across_patches_raises(self):
        patch_a = np.array([(0, 0, 1)], dtype=patch_dtype)
        patch_b = np.array([(0, 0, 1)], dtype=patch_dtype)  # same univ/address
        with self.assertRaises(ValueError):
            self.client.set_patches([patch_a, patch_b], [4, 4])

    def test_collision_within_single_patch_raises(self):
        patch = np.array([(0, 0, 1), (1, 0, 1)], dtype=patch_dtype)
        with self.assertRaises(ValueError):
            self.client.set_patch(patch)

    def test_failed_bind_leaves_previous_routing_intact(self):
        good = np.array([(0, 0, 1)], dtype=patch_dtype)
        self.client.set_patch(good)
        old_src = self.client._patch_src.copy()

        bad = np.array([(0, 0, 1)], dtype=patch_dtype)  # collides if merged
        with self.assertRaises(ValueError):
            self.client.set_patches([good, bad], [1, 1])

        np.testing.assert_array_equal(self.client._patch_src, old_src)
        self.client.set_patched_dmx_values(np.array([1.0]))
        self.assertEqual(self.client.buffer[0, 0], 255)

    def test_missing_frame_lengths_raises(self):
        patch_a = np.array([(0, 0, 1)], dtype=patch_dtype)
        patch_b = np.array([(0, 1, 1)], dtype=patch_dtype)
        with self.assertRaises(ValueError):
            self.client.set_patches([patch_a, patch_b])

    def test_mismatched_frame_lengths_raises(self):
        patch_a = np.array([(0, 0, 1)], dtype=patch_dtype)
        patch_b = np.array([(0, 1, 1)], dtype=patch_dtype)
        with self.assertRaises(ValueError):
            self.client.set_patches([patch_a, patch_b], [4])

    def test_non_positive_frame_length_raises(self):
        patch_a = np.array([(0, 0, 1)], dtype=patch_dtype)
        patch_b = np.array([(0, 1, 1)], dtype=patch_dtype)
        with self.assertRaises(ValueError):
            self.client.set_patches([patch_a, patch_b], [4, 0])

    def test_empty_patch_list_raises(self):
        with self.assertRaises(ValueError):
            self.client.set_patches([])

    def test_src_outside_declared_frame_raises(self):
        patch_a = np.array([(7, 0, 1)], dtype=patch_dtype)  # src 7 >= length 4
        patch_b = np.array([(0, 1, 1)], dtype=patch_dtype)
        with self.assertRaises(ValueError):
            self.client.set_patches([patch_a, patch_b], [4, 4])

    def test_invalid_addresses_masked_like_single_patch(self):
        patch_a = np.array([(0, 0, 0), (1, 0, 513), (2, 0, 1)], dtype=patch_dtype)
        patch_b = np.array([(0, 1, 1)], dtype=patch_dtype)
        self.client.set_patches([patch_a, patch_b], [4, 4])
        # Only (2, 0, 1) and (0, 1, 1) survive masking
        self.assertEqual(len(self.client._patch_src), 2)
        np.testing.assert_array_equal(self.client._patch_addr, [0, 0])

    def test_packet_size_grows_once_across_patches(self):
        client = ArtnetClient(target_ip="127.0.0.1", universes=[0], packet_size=2)
        patch_a = np.array([(0, 0, 100)], dtype=patch_dtype)
        patch_b = np.array([(0, 1, 300)], dtype=patch_dtype)
        client.set_patches([patch_a, patch_b], [4, 4])
        self.assertEqual(client.packet_size, 300)
        client.close()

    def test_packet_size_never_shrinks_on_rebind(self):
        patch_big = np.array([(0, 0, 512)], dtype=patch_dtype)
        patch_small = np.array([(0, 0, 2)], dtype=patch_dtype)
        self.client.set_patch(patch_big)
        self.client.set_patch(patch_small)
        self.assertEqual(self.client.packet_size, 512)

    def test_has_patch_initialized_false(self):
        fresh = ArtnetClient(target_ip="127.0.0.1")
        self.assertFalse(fresh.has_patch)
        # __str__ must not crash before any patch is registered
        str(fresh)
        fresh.close()


if __name__ == "__main__":
    unittest.main()
