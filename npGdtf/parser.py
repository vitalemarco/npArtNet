"""
GDTF parsing built entirely on the Python standard library.

A ``.gdtf`` file is a ZIP archive containing a ``description.xml`` document.
This module reads that archive with :mod:`zipfile` and parses the XML with
:mod:`xml.etree.ElementTree` — no third-party dependencies. It extracts the
fixture's DMX modes and channel byte offsets into the dataclasses defined in
:mod:`npGdtf.model`.
"""

import os
import xml.etree.ElementTree as ET
import zipfile

from .exceptions import GdtfError
from .model import DMXChannel, DMXMode, GdtfFixtureType

DESCRIPTION_FILENAME = "description.xml"


def load_gdtf(path: str | os.PathLike) -> GdtfFixtureType:
    """Load a ``.gdtf`` file and return its parsed fixture type.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a ``.gdtf`` archive on disk.

    Returns
    -------
    GdtfFixtureType
        The parsed fixture type and its DMX modes.

    Raises
    ------
    GdtfError
        If the file is not a valid ZIP archive, is missing
        ``description.xml``, or contains malformed XML.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = {name.lower(): name for name in archive.namelist()}
            real_name = names.get(DESCRIPTION_FILENAME)
            if real_name is None:
                raise GdtfError(
                    f"'{DESCRIPTION_FILENAME}' not found inside GDTF archive "
                    f"'{path}'."
                )
            xml_bytes = archive.read(real_name)
    except zipfile.BadZipFile as exc:
        raise GdtfError(f"'{path}' is not a valid GDTF (ZIP) archive.") from exc
    except FileNotFoundError as exc:
        raise GdtfError(f"GDTF file not found: '{path}'.") from exc

    return load_gdtf_from_xml(xml_bytes)


def load_gdtf_from_xml(xml: str | bytes) -> GdtfFixtureType:
    """Parse a GDTF ``description.xml`` document into a fixture type.

    Useful for testing or when the XML has already been extracted from the
    archive.

    Parameters
    ----------
    xml : str | bytes
        The contents of a GDTF ``description.xml`` document.

    Returns
    -------
    GdtfFixtureType
        The parsed fixture type and its DMX modes.

    Raises
    ------
    GdtfError
        If the XML is malformed or does not contain a ``FixtureType``.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise GdtfError(f"Malformed GDTF XML: {exc}") from exc

    data_version = root.get("DataVersion", "")

    fixture_el = root if root.tag == "FixtureType" else root.find("FixtureType")
    if fixture_el is None:
        raise GdtfError("GDTF document contains no <FixtureType> element.")

    modes = [
        _parse_mode(mode_el)
        for mode_el in fixture_el.findall("DMXModes/DMXMode")
    ]

    return GdtfFixtureType(
        name=fixture_el.get("Name", ""),
        short_name=fixture_el.get("ShortName", ""),
        manufacturer=fixture_el.get("Manufacturer", ""),
        modes=modes,
        data_version=data_version,
    )


def _parse_mode(mode_el: ET.Element) -> DMXMode:
    """Parse a single ``<DMXMode>`` element into a :class:`DMXMode`."""
    channels: list[DMXChannel] = []
    used_names: dict[str, int] = {}

    for ch_el in mode_el.findall("DMXChannels/DMXChannel"):
        channel = _parse_channel(ch_el)
        channel.attribute = _unique_name(channel.attribute, used_names)
        channels.append(channel)

    return DMXMode(
        name=mode_el.get("Name", ""),
        geometry=mode_el.get("Geometry"),
        channels=channels,
    )


def _parse_channel(ch_el: ET.Element) -> DMXChannel:
    """Parse a single ``<DMXChannel>`` element into a :class:`DMXChannel`."""
    offsets = _parse_offsets(ch_el.get("Offset"))
    num_bytes = len(offsets)

    attribute = _channel_attribute(ch_el)
    default = _channel_default(ch_el, num_bytes)

    return DMXChannel(
        attribute=attribute,
        offsets=offsets,
        default=default,
        geometry=ch_el.get("Geometry"),
        dmx_break=_parse_break(ch_el.get("DMXBreak")),
    )


def _parse_offsets(raw: str | None) -> list[int]:
    """Parse an ``Offset`` attribute (e.g. ``"1,2"``) into a list of ints.

    ``None``, an empty string, or the literal ``"None"`` yield an empty list,
    marking a virtual channel with no DMX footprint.
    """
    if not raw or raw.strip().lower() == "none":
        return []

    offsets: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            offsets.append(int(part))
        except ValueError:
            # Ignore unparsable tokens rather than failing the whole load.
            continue
    return offsets


def _parse_break(raw: str | None) -> int:
    """Parse a ``DMXBreak`` attribute, defaulting to 1."""
    if not raw:
        return 1
    try:
        return int(raw)
    except ValueError:
        return 1


def _channel_attribute(ch_el: ET.Element) -> str:
    """Derive a channel attribute name from its logical/function children."""
    logical = ch_el.find("LogicalChannel")
    if logical is not None:
        attr = logical.get("Attribute")
        if attr:
            return attr
        function = logical.find("ChannelFunction")
        if function is not None:
            attr = function.get("Attribute")
            if attr:
                return attr
    return "NoFeature"


def _channel_default(ch_el: ET.Element, num_bytes: int) -> int:
    """Read the channel's default DMX value at full channel bit-resolution.

    GDTF ``Default`` values use ``"value/bytecount"`` notation. The value is
    rescaled to the channel's own byte count so that, for example, a 16-bit
    channel whose default is declared as ``"128/1"`` resolves to ``128 << 8``.
    """
    logical = ch_el.find("LogicalChannel")
    if logical is None:
        return 0
    function = logical.find("ChannelFunction")
    if function is None:
        return 0

    value, byte_count = _parse_dmx_value(function.get("Default"))
    if num_bytes > byte_count:
        value <<= 8 * (num_bytes - byte_count)
    elif num_bytes and num_bytes < byte_count:
        value >>= 8 * (byte_count - num_bytes)
    return value


def _parse_dmx_value(raw: str | None) -> tuple[int, int]:
    """Parse a GDTF DMXValue ``"value/bytecount"`` string.

    Returns a ``(value, byte_count)`` tuple, defaulting to ``(0, 1)`` when the
    input is missing or unparsable.
    """
    if not raw:
        return (0, 1)
    text = raw.strip()
    value_part, _, byte_part = text.partition("/")
    try:
        value = int(value_part)
    except ValueError:
        return (0, 1)
    try:
        byte_count = int(byte_part) if byte_part else 1
    except ValueError:
        byte_count = 1
    return (value, max(byte_count, 1))


def _unique_name(name: str, used: dict[str, int]) -> str:
    """Return ``name`` made unique within a mode by suffixing duplicates."""
    count = used.get(name, 0) + 1
    used[name] = count
    if count == 1:
        return name
    return f"{name}_{count}"
