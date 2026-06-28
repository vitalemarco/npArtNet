"""Exceptions raised by the npGdtf package."""


class GdtfError(Exception):
    """Raised when a GDTF file cannot be read, parsed, or patched.

    Covers missing ``description.xml`` inside the archive, malformed XML,
    unknown DMX modes, and invalid patch requests (such as an unknown
    attribute name or a footprint that overflows a DMX universe).
    """
