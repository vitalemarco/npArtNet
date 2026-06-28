import io
import os
import tempfile
import unittest
import zipfile

from npGdtf import GdtfError, load_gdtf, load_gdtf_from_xml
from tests.gdtf_sample import (
    SAMPLE_DESCRIPTION_XML,
    build_sample_gdtf_bytes,
    write_sample_gdtf,
)


class TestGdtfParser(unittest.TestCase):
    def test_parse_fixture_metadata(self):
        ft = load_gdtf_from_xml(SAMPLE_DESCRIPTION_XML)
        self.assertEqual(ft.name, "Demo Mover")
        self.assertEqual(ft.short_name, "DM")
        self.assertEqual(ft.manufacturer, "ACME")
        self.assertEqual(ft.data_version, "1.1")
        self.assertEqual(ft.mode_names, ["Standard"])

    def test_parse_channels_and_offsets(self):
        mode = load_gdtf_from_xml(SAMPLE_DESCRIPTION_XML).get_mode("Standard")

        # Three declared channels, but only two occupy DMX bytes.
        self.assertEqual(len(mode.channels), 3)
        self.assertEqual(len(mode.physical_channels), 2)

        dimmer, pan = mode.physical_channels
        self.assertEqual(dimmer.attribute, "Dimmer")
        self.assertEqual(dimmer.offsets, [1])
        self.assertEqual(dimmer.num_bytes, 1)

        self.assertEqual(pan.attribute, "Pan")
        self.assertEqual(pan.offsets, [2, 3])  # MSB first
        self.assertEqual(pan.num_bytes, 2)

        # The virtual channel is parsed but has no footprint.
        control = mode.channels[2]
        self.assertEqual(control.attribute, "Control")
        self.assertTrue(control.is_virtual)

        # Footprint = highest byte offset.
        self.assertEqual(mode.footprint, 3)

    def test_default_rescaled_to_channel_resolution(self):
        mode = load_gdtf_from_xml(SAMPLE_DESCRIPTION_XML).get_mode("Standard")
        pan = mode.physical_channels[1]
        # "32768/2" is already 16-bit; stored as-is.
        self.assertEqual(pan.default, 32768)

    def test_load_from_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sample_gdtf(os.path.join(tmp, "demo.gdtf"))
            ft = load_gdtf(path)
        self.assertEqual(ft.name, "Demo Mover")
        self.assertEqual(len(ft.get_mode().physical_channels), 2)

    def test_sample_bytes_are_valid_zip(self):
        with zipfile.ZipFile(io.BytesIO(build_sample_gdtf_bytes())) as archive:
            self.assertIn("description.xml", archive.namelist())

    def test_get_mode_unknown_raises(self):
        ft = load_gdtf_from_xml(SAMPLE_DESCRIPTION_XML)
        with self.assertRaises(GdtfError):
            ft.get_mode("DoesNotExist")

    def test_missing_description_raises(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("other.txt", "nope")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.gdtf")
            with open(path, "wb") as handle:
                handle.write(buffer.getvalue())
            with self.assertRaises(GdtfError):
                load_gdtf(path)

    def test_malformed_xml_raises(self):
        with self.assertRaises(GdtfError):
            load_gdtf_from_xml("<GDTF><FixtureType></GDTF>")

    def test_duplicate_attribute_names_made_unique(self):
        xml = """<?xml version="1.0"?>
        <GDTF DataVersion="1.1">
          <FixtureType Name="Dup" ShortName="D" Manufacturer="ACME">
            <DMXModes>
              <DMXMode Name="M">
                <DMXChannels>
                  <DMXChannel Offset="1">
                    <LogicalChannel Attribute="Dimmer"/>
                  </DMXChannel>
                  <DMXChannel Offset="2">
                    <LogicalChannel Attribute="Dimmer"/>
                  </DMXChannel>
                </DMXChannels>
              </DMXMode>
            </DMXModes>
          </FixtureType>
        </GDTF>"""
        mode = load_gdtf_from_xml(xml).get_mode("M")
        self.assertEqual([c.attribute for c in mode.channels], ["Dimmer", "Dimmer_2"])


if __name__ == "__main__":
    unittest.main()
