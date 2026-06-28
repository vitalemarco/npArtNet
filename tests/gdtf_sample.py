"""Helpers that build a synthetic in-memory GDTF for the npGdtf tests.

Keeping the sample fixture in code (rather than committing a binary ``.gdtf``)
means the tests have no external assets and document exactly what they expect.
"""

import io
import zipfile

# A minimal but realistic fixture: an 8-bit Dimmer, a 16-bit Pan (coarse/fine),
# and a virtual control channel with no DMX footprint.
SAMPLE_DESCRIPTION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<GDTF DataVersion="1.1">
  <FixtureType Name="Demo Mover" ShortName="DM" Manufacturer="ACME">
    <DMXModes>
      <DMXMode Name="Standard" Geometry="Base">
        <DMXChannels>
          <DMXChannel DMXBreak="1" Offset="1" Geometry="Body">
            <LogicalChannel Attribute="Dimmer">
              <ChannelFunction Attribute="Dimmer" Default="0/1"/>
            </LogicalChannel>
          </DMXChannel>
          <DMXChannel DMXBreak="1" Offset="2,3" Geometry="Yoke">
            <LogicalChannel Attribute="Pan">
              <ChannelFunction Attribute="Pan" Default="32768/2"/>
            </LogicalChannel>
          </DMXChannel>
          <DMXChannel DMXBreak="1" Offset="None">
            <LogicalChannel Attribute="Control">
              <ChannelFunction Attribute="Control" Default="0/1"/>
            </LogicalChannel>
          </DMXChannel>
        </DMXChannels>
      </DMXMode>
    </DMXModes>
  </FixtureType>
</GDTF>
"""


def build_sample_gdtf_bytes() -> bytes:
    """Return the bytes of a valid ``.gdtf`` (ZIP) archive in memory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("description.xml", SAMPLE_DESCRIPTION_XML)
    return buffer.getvalue()


def write_sample_gdtf(path: str) -> str:
    """Write the sample ``.gdtf`` archive to ``path`` and return the path."""
    with open(path, "wb") as handle:
        handle.write(build_sample_gdtf_bytes())
    return path
