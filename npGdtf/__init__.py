"""
npGdtf
======

GDTF fixture patching for `npArtNet`, with **no external dependencies**.

`npArtNet` is the "dumb math" layer: it maps a flat array of normalized floats
straight into DMX universes via a ``patch_dtype`` map. `npGdtf` is the missing
"smart" layer on top: it loads real fixture definitions from ``.gdtf`` files
(parsed with the standard library only) and lets you patch them onto universes
and DMX addresses, generating the ``patch_dtype`` map for you and handling
16/24-bit channel splits automatically.

A ``.gdtf`` file is a ZIP archive containing a ``description.xml``. `npGdtf`
reads the fixture's DMX modes and each channel's byte offsets, then expands a
patched fixture into one source slot per DMX byte so it plugs directly into
``ArtnetClient``.

Example
-------

```python
from npArtNet import ArtnetClient
from npGdtf import load_gdtf, Rig

# 1. Load fixture definitions from .gdtf files (stdlib zip + XML parsing)
mover = load_gdtf("MovingHead.gdtf")
par = load_gdtf("LedPar.gdtf")

# 2. Patch fixtures onto universes / DMX addresses
rig = Rig()
mh = rig.patch(mover, mode="Standard", universe=0, address=1)
p1 = rig.patch(par, universe=0, address=20)   # first mode by default

# 3. Compile the layout and register it with the client
client = ArtnetClient(target_ip="127.0.0.1")
client.set_patch(rig.build_patch_map())

# 4. Drive fixtures by attribute name (normalized 0.0 - 1.0)
mh.set("Pan", 0.5)      # 16-bit attributes are split coarse/fine for you
mh.set("Dimmer", 1.0)
p1.set("ColorAdd_R", 1.0)

client.set_patched_dmx_values(rig.get_state())
client.send_package()
```

See ``examples/4.1_gdtf_patching.py`` for a complete, runnable demo.
"""

from .exceptions import GdtfError
from .fixture import PatchedFixture, Rig
from .model import DMXChannel, DMXMode, GdtfFixtureType
from .parser import load_gdtf, load_gdtf_from_xml

__all__ = [
    "load_gdtf",
    "load_gdtf_from_xml",
    "GdtfFixtureType",
    "DMXMode",
    "DMXChannel",
    "PatchedFixture",
    "Rig",
    "GdtfError",
]
