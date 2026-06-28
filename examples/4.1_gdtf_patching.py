"""
Example 4.1 — Patching fixtures from GDTF files with npGdtf.

This demonstrates the full npGdtf flow without needing a vendor ``.gdtf`` file:
a small fixture definition is written to a temporary ``.gdtf`` archive, loaded
with ``load_gdtf``, patched onto universes with ``Rig``, and then driven by
attribute name into ``npArtNet``.

Run a matching server (e.g. ``examples/1.x``) on universes 0 and 1 to watch the
output, or just run this standalone to see it transmit.
"""

import io
import math
import os
import tempfile
import time
import zipfile

from npArtNet import ArtnetClient
from npGdtf import load_gdtf, Rig

# A 4-channel RGB+Dimmer fixture and a 16-bit pan/tilt mover, described inline.
FIXTURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<GDTF DataVersion="1.1">
  <FixtureType Name="Demo Mover" ShortName="DM" Manufacturer="ACME">
    <DMXModes>
      <DMXMode Name="Standard" Geometry="Base">
        <DMXChannels>
          <DMXChannel Offset="1"><LogicalChannel Attribute="Pan">
            <ChannelFunction Attribute="Pan" Default="32768/2"/></LogicalChannel></DMXChannel>
          <DMXChannel Offset="2,3"><LogicalChannel Attribute="Tilt">
            <ChannelFunction Attribute="Tilt" Default="32768/2"/></LogicalChannel></DMXChannel>
          <DMXChannel Offset="4"><LogicalChannel Attribute="Dimmer">
            <ChannelFunction Attribute="Dimmer" Default="0/1"/></LogicalChannel></DMXChannel>
        </DMXChannels>
      </DMXMode>
    </DMXModes>
  </FixtureType>
</GDTF>
"""


def make_sample_gdtf(path: str) -> str:
    """Write a tiny valid ``.gdtf`` archive so the example is self-contained."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("description.xml", FIXTURE_XML)
    with open(path, "wb") as handle:
        handle.write(buffer.getvalue())
    return path


def main():
    with tempfile.TemporaryDirectory() as tmp:
        gdtf_path = make_sample_gdtf(os.path.join(tmp, "demo_mover.gdtf"))

        # 1. Load the fixture definition straight from the .gdtf file.
        mover = load_gdtf(gdtf_path)
        print(
            f"Loaded '{mover.name}' by {mover.manufacturer} "
            f"(modes: {mover.mode_names})"
        )

        # 2. Patch two of them onto the rig.
        rig = Rig()
        mh1 = rig.patch(mover, mode="Standard", universe=0, address=1)
        mh2 = rig.patch(mover, mode="Standard", universe=1, address=1)

        # 3. Compile the layout and hand it to the client.
        client = ArtnetClient(target_ip="127.0.0.1")
        client.set_patch(rig.build_patch_map())
        print(
            f"Patched {len(rig.fixtures)} fixtures across "
            f"{len(client.universes)} universe(s). Attributes: {mh1.attributes}"
        )
        print("Transmitting. Press Ctrl+C to stop.")

    start = time.time()
    try:
        while True:
            t = time.time() - start

            # 4. Drive fixtures by attribute name (normalized 0.0 - 1.0).
            #    16-bit Tilt is split into coarse/fine bytes automatically.
            mh1.set("Pan", (math.sin(t) + 1.0) / 2.0)
            mh1.set("Tilt", (math.cos(t) + 1.0) / 2.0)
            mh1.set("Dimmer", 1.0)

            mh2.set("Pan", (math.cos(t) + 1.0) / 2.0)
            mh2.set("Tilt", (math.sin(t) + 1.0) / 2.0)
            mh2.set("Dimmer", (math.sin(t * 3.0) + 1.0) / 2.0)

            client.set_patched_dmx_values(rig.get_state())
            client.send_package()
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
