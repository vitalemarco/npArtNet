"""
The patching engine that turns parsed GDTF fixtures into an npArtNet patch.

:class:`Rig` mirrors the ``SimpleEngine`` pattern from
``examples/3.1_simple_engine.py``: it allocates a slice of a flat ``float32``
state array to each patched fixture, builds the combined ``patch_dtype`` map for
``ArtnetClient.set_patch``, and lets you drive fixtures by attribute name.

A fixture's DMX footprint is expanded to **one source slot per DMX byte** so it
plugs straight into npArtNet's one-float-per-byte model. Multi-byte (16/24-bit)
attributes are split big-endian across their slots, so npArtNet's ``* 255``
scaling reproduces exactly the right coarse/fine bytes.
"""

import numpy as np

from npArtNet.data_types import DMX_UNIVERSE_SIZE, patch_dtype
from npArtNet.utils import get_msb_lsb  # noqa: F401  (re-exported for callers)

from .exceptions import GdtfError
from .model import DMXMode, GdtfFixtureType


def _clamp01(value: float) -> float:
    """Bound a value to the normalized ``[0.0, 1.0]`` range."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class PatchedFixture:
    """A fixture instance placed at a universe and DMX start address.

    Instances are created by :meth:`Rig.patch`, not directly. After the rig's
    patch map is built, set attribute values with :meth:`set` (normalized
    ``0.0``-``1.0``) and the rig's state array is updated in place.
    """

    def __init__(
        self,
        rig: "Rig",
        fixture_type: GdtfFixtureType,
        mode: DMXMode,
        universe: int,
        address: int,
        src_start: int,
    ):
        self.fixture_type = fixture_type
        self.mode = mode
        self.universe = int(universe)
        self.address = int(address)
        self.src_start = int(src_start)
        self._rig = rig

        # attribute name -> ordered src slots (most-significant byte first)
        self.attribute_map: dict[str, list[int]] = {}
        # attribute name -> bit-resolution default value
        self._defaults: dict[str, int] = {}
        self._patch_rows: list[tuple[int, int, int]] = []

        self._build()

    @property
    def num_slots(self) -> int:
        """Number of source slots (DMX bytes) this fixture occupies."""
        return len(self._patch_rows)

    def _build(self) -> None:
        """Expand the mode into per-byte src slots and patch rows."""
        byte_index = 0
        for channel in self.mode.physical_channels:
            slots: list[int] = []
            for offset in channel.offsets:
                dmx_address = self.address + offset - 1
                if dmx_address < 1 or dmx_address > DMX_UNIVERSE_SIZE:
                    raise GdtfError(
                        f"Fixture '{self.fixture_type.name}' channel "
                        f"'{channel.attribute}' maps to DMX address "
                        f"{dmx_address}, outside 1..{DMX_UNIVERSE_SIZE}. "
                        f"Check the start address ({self.address})."
                    )
                src = self.src_start + byte_index
                self._patch_rows.append((src, self.universe, dmx_address))
                slots.append(src)
                byte_index += 1
            self.attribute_map[channel.attribute] = slots
            self._defaults[channel.attribute] = channel.default

    def get_patch(self) -> list[tuple[int, int, int]]:
        """Return this fixture's ``(src, universe, address)`` patch rows."""
        return list(self._patch_rows)

    @property
    def attributes(self) -> list[str]:
        """Names of the controllable attributes on this fixture."""
        return list(self.attribute_map)

    def set(self, attribute: str, value: float) -> None:
        """Set an attribute from a normalized ``0.0``-``1.0`` value.

        The value is scaled to the attribute's full bit-resolution and split
        big-endian across its DMX bytes.

        Parameters
        ----------
        attribute : str
            The attribute name (see :attr:`attributes`).
        value : float
            Normalized intensity/position in ``[0.0, 1.0]``.

        Raises
        ------
        GdtfError
            If the attribute is unknown to this fixture.
        """
        slots = self._slots_for(attribute)
        n = len(slots)
        max_value = (1 << (8 * n)) - 1
        raw = int(round(_clamp01(value) * max_value))
        self._write_bytes(slots, raw)

    def set_raw(self, attribute: str, dmx_value: int) -> None:
        """Set an attribute from a raw DMX value at full bit-resolution.

        For an 8-bit channel this is ``0``-``255``; for a 16-bit channel it is
        ``0``-``65535``; and so on. The value is clamped to the valid range.
        """
        slots = self._slots_for(attribute)
        n = len(slots)
        max_value = (1 << (8 * n)) - 1
        raw = max(0, min(int(dmx_value), max_value))
        self._write_bytes(slots, raw)

    def _slots_for(self, attribute: str) -> list[int]:
        slots = self.attribute_map.get(attribute)
        if slots is None:
            raise GdtfError(
                f"Unknown attribute '{attribute}' on fixture "
                f"'{self.fixture_type.name}'. Available: {self.attributes}"
            )
        return slots

    def _write_bytes(self, slots: list[int], raw: int) -> None:
        """Split ``raw`` big-endian across ``slots`` and write to rig state."""
        state = self._rig.state
        if state is None:
            raise GdtfError(
                "Rig patch map not built yet. Call Rig.build_patch_map() "
                "before setting fixture values."
            )
        n = len(slots)
        for i, src in enumerate(slots):
            shift = 8 * (n - 1 - i)  # most-significant byte first
            byte = (raw >> shift) & 0xFF
            state[src] = byte / 255.0


class Rig:
    """A collection of patched fixtures sharing one npArtNet state array.

    Patch fixtures with :meth:`patch`, compile the layout with
    :meth:`build_patch_map` (hand the result to ``ArtnetClient.set_patch``),
    then drive fixtures by attribute name and feed :meth:`get_state` to
    ``ArtnetClient.set_patched_dmx_values`` each frame.
    """

    def __init__(self):
        self.fixtures: list[PatchedFixture] = []
        self.total_slots: int = 0
        self.state: np.ndarray | None = None

    def patch(
        self,
        fixture_type: GdtfFixtureType,
        mode: str | DMXMode | None = None,
        universe: int = 0,
        address: int = 1,
    ) -> PatchedFixture:
        """Place a fixture in DMX address space and register it with the rig.

        Parameters
        ----------
        fixture_type : GdtfFixtureType
            A fixture type loaded via :func:`npGdtf.load_gdtf`.
        mode : str | DMXMode | None
            The DMX mode to use, by name or instance. Defaults to the
            fixture's first mode.
        universe : int
            The Art-Net universe to patch into.
        address : int
            The 1-based DMX start address.

        Returns
        -------
        PatchedFixture
            The newly patched fixture.
        """
        resolved = mode if isinstance(mode, DMXMode) else fixture_type.get_mode(mode)

        fixture = PatchedFixture(
            rig=self,
            fixture_type=fixture_type,
            mode=resolved,
            universe=universe,
            address=address,
            src_start=self.total_slots,
        )
        self.total_slots += len(fixture.get_patch())
        self.fixtures.append(fixture)
        # Invalidate any previously built state; the layout changed.
        self.state = None
        return fixture

    def build_patch_map(self) -> np.ndarray:
        """Compile all fixtures into an npArtNet ``patch_dtype`` array.

        Allocates the shared ``float32`` state array (sized to the total
        number of source slots) and applies each fixture's GDTF default values.

        Returns
        -------
        np.ndarray
            A structured array (``patch_dtype``) ready for
            ``ArtnetClient.set_patch``.
        """
        self.state = np.zeros(self.total_slots, dtype=np.float32)

        rows: list[tuple[int, int, int]] = []
        for fixture in self.fixtures:
            rows.extend(fixture.get_patch())

        # Apply GDTF defaults now that the state array exists.
        for fixture in self.fixtures:
            for attribute, default in fixture._defaults.items():
                if default:
                    fixture.set_raw(attribute, default)

        if not rows:
            return np.empty(0, dtype=patch_dtype)
        return np.array(rows, dtype=patch_dtype)

    def get_state(self) -> np.ndarray:
        """Return the shared normalized state array for transmission.

        Raises
        ------
        GdtfError
            If :meth:`build_patch_map` has not been called yet.
        """
        if self.state is None:
            raise GdtfError(
                "Rig patch map not built yet. Call Rig.build_patch_map() first."
            )
        return self.state
