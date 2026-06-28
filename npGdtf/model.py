"""
Data model for parsed GDTF fixture definitions.

These dataclasses describe only the subset of GDTF needed to lay a fixture out
in DMX address space: the DMX modes, their channels, and each channel's byte
offsets within the fixture footprint. Higher-level GDTF concepts (channel
functions, physical ranges, wheels, geometry trees) are intentionally omitted;
see the package docstring for the rationale.
"""

from dataclasses import dataclass, field

from .exceptions import GdtfError


@dataclass
class DMXChannel:
    """A single DMX channel within a mode.

    Parameters
    ----------
    attribute : str
        Human-readable attribute name (e.g. ``"Dimmer"``, ``"Pan"``). Derived
        from the GDTF ``LogicalChannel``/``ChannelFunction`` attribute, made
        unique within a mode by the parser.
    offsets : list[int]
        1-based byte position(s) of this channel within the fixture footprint,
        most-significant first. One offset is an 8-bit channel; two offsets are
        a 16-bit coarse/fine pair, and so on. An empty list marks a virtual
        channel with no DMX footprint.
    default : int
        Default DMX value for the channel, expressed at the channel's full
        bit-resolution (e.g. 0..65535 for a 16-bit channel).
    geometry : str | None
        The geometry the channel is attached to, if declared.
    dmx_break : int
        The DMX break the channel belongs to (defaults to 1).
    """

    attribute: str
    offsets: list[int] = field(default_factory=list)
    default: int = 0
    geometry: str | None = None
    dmx_break: int = 1

    @property
    def num_bytes(self) -> int:
        """Number of DMX bytes occupied by this channel (0 for virtual)."""
        return len(self.offsets)

    @property
    def is_virtual(self) -> bool:
        """True when the channel has no DMX footprint."""
        return not self.offsets


@dataclass
class DMXMode:
    """A named DMX mode (a specific channel layout) of a fixture type.

    Parameters
    ----------
    name : str
        The mode name as declared in the GDTF (e.g. ``"Standard"``).
    geometry : str | None
        The root geometry the mode is bound to, if declared.
    channels : list[DMXChannel]
        All channels of the mode, in declaration order.
    """

    name: str
    geometry: str | None = None
    channels: list[DMXChannel] = field(default_factory=list)

    @property
    def physical_channels(self) -> list[DMXChannel]:
        """Channels that occupy DMX bytes (virtual channels excluded)."""
        return [ch for ch in self.channels if not ch.is_virtual]

    @property
    def footprint(self) -> int:
        """Number of DMX addresses the mode occupies (highest byte offset).

        Returns 0 when the mode has no physical channels.
        """
        max_offset = 0
        for ch in self.channels:
            for off in ch.offsets:
                if off > max_offset:
                    max_offset = off
        return max_offset


@dataclass
class GdtfFixtureType:
    """A parsed GDTF fixture type and its available DMX modes.

    Parameters
    ----------
    name : str
        Long fixture name.
    short_name : str
        Abbreviated fixture name.
    manufacturer : str
        Manufacturer name.
    modes : list[DMXMode]
        Available DMX modes.
    data_version : str
        The GDTF ``DataVersion`` declared by the file.
    """

    name: str
    short_name: str = ""
    manufacturer: str = ""
    modes: list[DMXMode] = field(default_factory=list)
    data_version: str = ""

    @property
    def mode_names(self) -> list[str]:
        """The names of all available DMX modes."""
        return [m.name for m in self.modes]

    def get_mode(self, name: str | None = None) -> DMXMode:
        """Return a DMX mode by name, or the first mode if ``name`` is None.

        Parameters
        ----------
        name : str | None
            The mode name to look up. When None, the first declared mode is
            returned.

        Returns
        -------
        DMXMode
            The matching mode.

        Raises
        ------
        GdtfError
            If the fixture has no modes, or the requested name is unknown.
        """
        if not self.modes:
            raise GdtfError(f"Fixture '{self.name}' declares no DMX modes.")
        if name is None:
            return self.modes[0]
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise GdtfError(
            f"Unknown DMX mode '{name}' for fixture '{self.name}'. "
            f"Available modes: {self.mode_names}"
        )
