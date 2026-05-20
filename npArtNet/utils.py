"""Provides common bitwise and mathematical utility functions for Art-Net."""


def get_msb_lsb(number: int, high_first: bool = True) -> tuple[int, int]:
    """
    Extract the Most Significant Byte (MSB) and Least Significant Byte (LSB) from an integer.

    Parameters
    ----------
    number : int
        The integer value to split into bytes.
    high_first : bool, optional
        If True, returns the tuple as (MSB, LSB). If False, returns (LSB, MSB).
        Default is True.

    Returns
    -------
    tuple[int, int]
        The separated byte values.
    """

    low = number & 0xFF
    high = (number >> 8) & 0xFF

    if high_first:
        return (high, low)
    return (low, high)


def clamp_value(
    number: int, range_min: int, range_max: int, make_even: bool = False
) -> int:
    """
    Bound a number to a specific range, optionally rounding up to the nearest even number.

    Parameters
    ----------
    number : int
        The value to evaluate and constrain.
    range_min : int
        The lowest allowed value.
    range_max : int
        The highest allowed value.
    make_even : bool, optional
        If True, forces the resulting number to be even by adding 1 if it is odd.
        Default is False.

    Returns
    -------
    int
        The bounded (and optionally even) integer.
    """

    number = max(range_min, min(number, range_max))

    if make_even and number % 2 != 0:
        number += 1

    return number


def make_address_mask(
    universe: int, sub: int = 0, net: int = 0, is_simplified: bool = True
) -> bytearray:
    """
    Generate the 2-byte address mask for a given universe, subnet, and net.

    Parameters
    ----------
    universe : int
        The universe to map (0-32767 for simplified, 0-15 for standard).
    sub : int, optional
        The subnet to map (0-15). Ignored if is_simplified is True. Default is 0.
    net : int, optional
        The net to map (0-127). Ignored if is_simplified is True. Default is 0.
    is_simplified : bool, optional
        Whether to use simplified 15-bit universe mapping. Default is True.

    Returns
    -------
    bytearray
        A 2-byte array representing the Art-Net address mask for routing.
    """

    address_mask = bytearray()

    if is_simplified:
        universe = clamp_value(universe, 0, 32767)
        msb, lsb = get_msb_lsb(universe)
        address_mask.extend([lsb, msb])
    else:
        universe = clamp_value(universe, 0, 15)
        sub = clamp_value(sub, 0, 15)
        net = clamp_value(net, 0, 127)

        # BITWISE SHIFT TO COMBINE SUBNET (TOP 4 BITS) AND UNIVERSE (BOTTOM 4 BITS)
        address_mask.append(sub << 4 | universe)
        address_mask.append(net & 0xFF)

    return address_mask
