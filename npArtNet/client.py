"""High-performance, NumPy-backed implementation of an Art-Net Client."""

import numpy as np
import socket
from .data_types import DMX_UNIVERSE_SIZE
from .utils import get_msb_lsb, clamp_value


# --------------------------------
# CLIENT
# --------------------------------
class ArtnetClient:
    """
    A vectorized, multi-universe Art-Net sender backed by NumPy.

    Maintains contiguous memory buffers for ultra-fast manipulation and transmission
    of massive lighting matrices. Supports integrated patching and dynamic packet sizing.

    Parameters
    ----------
    target_ip : str, optional
        The IP address of the destination node or broadcast address. Default is "127.0.0.1".
    universes : list[int] | None, optional
        A list of initial universes to register. Default is [0].
    packet_size : int, optional
        The default payload size in bytes. Default is 512.
    even_packet_size : bool, optional
        If True, forces all packet sizes to be even numbers per Art-Net spec. Default is True.
    broadcast : bool, optional
        If True, enables UDP broadcast mode on the socket. Default is False.
    source_address : str | None, optional
        Optional specific IP address to bind the outgoing socket to. Default is None.
    artsync : bool, optional
        If True, transmits an ArtSync packet after a full frame transmission. Default is True.
    port : int, optional
        The UDP port to transmit to. Default is 6454.
    """

    def __init__(
        self,
        target_ip: str = "127.0.0.1",
        universes: list[int] | None = None,
        packet_size: int = DMX_UNIVERSE_SIZE,
        even_packet_size: bool = True,
        broadcast: bool = False,
        source_address: str | None = None,
        artsync: bool = True,
        port: int = 6454,
    ):

        # ROUTING
        self.target_ip: str = target_ip
        self.port: int = port

        # UNIVERSE MAPPING
        if universes is None:
            universes = [0]
        self.universes: list[int] = list(universes)
        self.universe_map: dict[int, int] = {u: i for i, u in enumerate(self.universes)}
        self.num_universes: int = len(self.universes)

        # ArtNet - NETWORK HIERACHY
        self.is_simplified: bool = True
        self.subnet: int = 0
        self.net: int = 0

        # DATA SIZE
        self.make_even: bool = even_packet_size
        self.packet_size: int = clamp_value(
            packet_size, 2, DMX_UNIVERSE_SIZE, self.make_even
        )

        # NUMPY BUFFERS
        self.buffer: np.ndarray = np.zeros(
            (self.num_universes, self.packet_size), dtype=np.uint8
        )
        self.sequences: np.ndarray = np.zeros(self.num_universes, dtype=np.uint8)

        # PRE-BUILD HEADERS
        self.headers: list[bytearray] = [
            bytearray(self._build_header(self.universes[i]))
            for i in range(self.num_universes)
        ]

        # PRE-BUILD PACKETS
        self.packets: list[bytearray] = [
            bytearray(self.headers[i]) + bytearray(self.packet_size)
            for i in range(self.num_universes)
        ]

        # PATCH ROUTING (EMPTY UNTIL set_patch() / set_patches() IS CALLED)
        self.has_patch: bool = False
        self._patch_frame_lengths: list[int] | None = None

        # SYNCING MULTIPLE PACKAGES
        self.sync = artsync
        if self.sync:
            self.artsync_header: bytearray = self._build_artsync_header()

        # SOCKET
        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if broadcast:
            self.socket_client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        if source_address:
            self.socket_client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket_client.bind(source_address)

    # --------------------------------
    # HEADERS
    # --------------------------------
    def _build_header(self, universe_value: int) -> bytearray:
        """Build the ArtDMX packet header.

        Parameters
        ----------
        universe_value : int
            The universe value for the header.

        Returns
        -------
        bytearray
            The ArtDMX packet header.
        """

        header: bytearray = bytearray()

        # 0 - ID (7 BYTES + Null)
        header.extend(bytearray("Art-Net", "utf8"))
        header.append(0x0)

        # 8 - OPCODE (2 x 8 LOW BYTE FIRST)
        header.append(0x00)
        header.append(0x50)  # ArtDMX DATA PACKET

        # 10 - PROTOCOL VERSION (2 x 8 HIGH BYTE FIRST)
        header.append(0x0)
        header.append(14)

        # 12 - SEQUENCE (INT8), PLACEHOLDER (GETS OVERWRITTEN IN send_package())
        header.append(0x00)

        # 13 - PHYSICAL PORT (INT8)
        header.append(0x00)

        # 14 - UNIVERSE, SUBNET, NET LOGIC
        if self.is_simplified:
            msb, lsb = get_msb_lsb(universe_value)
            header.append(lsb)
            header.append(msb)
        else:
            # IN STANDARD ArtNet 4, THE UNIVERSE VALUE IS 0-15 PER SUBNET
            uni_bounded = clamp_value(universe_value, 0, 15, False)
            header.append(self.subnet << 4 | uni_bounded)
            header.append(self.net & 0xFF)

        # 16 - PACKET SIZE (2 x 8 HIGH BYTE FIRST)
        msb, lsb = get_msb_lsb(self.packet_size)
        header.append(msb)
        header.append(lsb)

        return header

    def _build_artsync_header(self) -> bytearray:
        """Build the ArtSync packet header.

        Returns
        -------
        bytearray
            The ArtSync packet header.
        """

        artsync_header: bytearray = bytearray()

        # ID: ARRAY OF 8 CHARACTERS, THE FINAL CHARACTER IS A NULL TERMINATION.
        artsync_header.extend(bytearray("Art-Net", "utf8"))
        artsync_header.append(0x0)

        # OPCODE: DEFINES THE CLASS OF DATA WITHIN THIS UDP PACKET. TRANSMITTED LOW BYTE FIRST.
        artsync_header.append(0x00)
        artsync_header.append(0x52)

        # PROTVERHI AND PROTVERLO: ArtNet PROTOCOL REVISION NUMBER. CURRENT REVISION = 14
        artsync_header.append(0x0)
        artsync_header.append(14)

        # AUX1 AND AUX2: TRANSMITTED AS ZERO.
        artsync_header.append(0x0)
        artsync_header.append(0x0)

        return artsync_header

    def _rebuild_headers(self):
        """Rebuild all headers and packets based on the current configuration."""

        # REBUILD ALL HEADERS
        self.headers = [
            bytearray(self._build_header(self.universes[i]))
            for i in range(self.num_universes)
        ]

        # REBUILD ALL PACKETS
        self.packets = [
            bytearray(self.headers[i]) + bytearray(self.packet_size)
            for i in range(self.num_universes)
        ]

    # --------------------------------
    # SEND PACKAGES
    # --------------------------------
    def send_artsync(self):
        """Transmit a single ArtSync packet to synchronize nodes."""

        try:
            self.socket_client.sendto(self.artsync_header, (self.target_ip, self.port))
        except socket.error as error:
            print(f"ERROR: Socket error with exception: {error}")

    def send_package(self):
        """
        Blast the current internal buffer state to the network.

        Iterates through the registered universes, appends sequence numbers,
        concatenates headers and payloads, and sends the UDP packets.
        Transmits an ArtSync packet at the end if enabled.
        """

        for i in range(self.num_universes):
            self.packets[i][12] = self.sequences[i]
            self.packets[i][18:] = memoryview(self.buffer[i])

            try:
                self.socket_client.sendto(self.packets[i], (self.target_ip, self.port))
            except socket.error as error:
                print(f"ERROR: Socket error with exception: {error}")

            sequence = int(self.sequences[i]) + 1
            if sequence > 255:
                sequence = 1
            self.sequences[i] = sequence

        if self.sync:
            self.send_artsync()

    # --------------------------------
    # SET CLIENT DATA
    # --------------------------------
    def set_subnet(self, sub: int):
        """
        Set the routing subnet and rebuild the packet headers.

        Parameters
        ----------
        sub : int
            The subnet address (0 to 15).
        """

        self.subnet = clamp_value(sub, 0, 15, False)
        self._rebuild_headers()

    def set_net(self, net: int):
        """
        Set the routing net and rebuild the packet headers.

        Parameters
        ----------
        net : int
            The net address (0 to 127).
        """

        self.net = clamp_value(net, 0, 127, False)
        self._rebuild_headers()

    def set_simplified(self, is_simplified: bool):
        """
        Toggle simplified 15-bit universe addressing and rebuild headers.

        Parameters
        ----------
        is_simplified : bool
            True for 15-bit simplified addressing, False for standard Net/Sub/Uni.
        """

        self.is_simplified = is_simplified
        self._rebuild_headers()

    def set_packet_size(self, packet_size: int):
        """
        Dynamically resize the transmission buffer's column width.

        Resizes the internal matrix while preserving existing data,
        and rebuilds headers to reflect the new byte length.

        Parameters
        ----------
        packet_size : int
            The requested size (2 to 512). Will be forced even if `make_even` is True.
        """

        new_size: int = clamp_value(packet_size, 2, DMX_UNIVERSE_SIZE, self.make_even)

        if new_size != self.packet_size:
            self.packet_size = new_size
            old_buffer = self.buffer
            self.buffer = np.zeros(
                (self.num_universes, self.packet_size), dtype=np.uint8
            )

            copy_size = min(old_buffer.shape[1], self.packet_size)
            self.buffer[:, :copy_size] = old_buffer[:, :copy_size]

            self._rebuild_headers()

    def add_universe(self, universe: int) -> int:
        """
        Dynamically register a new universe and append a new row to the internal matrix.

        Parameters
        ----------
        universe : int
            The new universe number to register.

        Returns
        -------
        int
            The internal row index (0-based) assigned to the added universe.
        """

        if universe in self.universe_map:
            return self.universe_map[universe]

        self.universes.append(universe)
        row_idx = len(self.universes) - 1
        self.universe_map[universe] = row_idx
        self.num_universes += 1

        new_row = np.zeros((1, self.packet_size), dtype=np.uint8)
        self.buffer = np.vstack((self.buffer, new_row))
        self.sequences = np.append(self.sequences, 0)

        new_header = bytearray(self._build_header(universe))
        self.headers.append(new_header)
        self.packets.append(new_header + bytearray(self.packet_size))

        return row_idx

    def set_patches(
        self,
        patch_maps: list[np.ndarray],
        frame_lengths: list[int] | None = None,
    ):
        """
        Bind multiple source patches into a single merged routing.

        Merges all patch maps into one O(1) routing by shifting each source's
        ``src`` indices by the cumulative length of the preceding sources' frames.
        Universes are registered once, packet sizes are grown once across all
        patches, and routing indices are built once. This is configuration-time
        work; per frame it costs a single indexed write.

        Multi-Source Contract
        ---------------------
        Bind patches in order ``P0 .. Pn`` with frame lengths ``L0 .. Ln``.
        Each frame, pass the concatenation of all source frames in patch order::

            client.set_patched_dmx_values(np.concatenate([frame0, ..., framen]))
            client.send_package()

        Each ``framei`` must contain exactly ``Li`` normalized floats (0.0 to 1.0).
        One indexed write routes everything; one ``send_package()`` bursts all
        universes. Note that a source's frame space is fully reserved by its
        declared length ``Li``, even if parts of the canvas are unpatched:
        offsets are derived from ``frame_lengths``, never from the patch maps.

        Parameters
        ----------
        patch_maps : list[np.ndarray]
            A list of structured arrays (patch_dtype), one per frame source.
        frame_lengths : list[int] | None, optional
            The flat frame length of each source, in the same order as
            ``patch_maps``. Required when binding more than one patch, since a
            source canvas can be larger than its wired patch and therefore the
            frame length cannot be derived from the patch map. May be omitted
            (None) only for a single patch. Default is None.

        Raises
        ------
        ValueError
            If no patch maps are given, if ``frame_lengths`` is missing or
            mismatched for multiple patches, if a patch references a ``src``
            index outside its declared frame length, or if duplicate
            ``(universe, address)`` pairs exist across the merged routing.

        Examples
        --------
        >>> client.set_patches([zone_a_patch, zone_b_patch], [768, 768])
        >>> frame = np.concatenate([zone_a_frame, zone_b_frame])
        >>> client.set_patched_dmx_values(frame)
        >>> client.send_package()
        """

        if len(patch_maps) == 0:
            raise ValueError("set_patches requires at least one patch map.")

        # VALIDATE FRAME LENGTHS AND PRECOMPUTE SOURCE OFFSETS
        if frame_lengths is None:
            if len(patch_maps) > 1:
                raise ValueError(
                    "frame_lengths is required when binding multiple patches: "
                    "a source canvas can be larger than its wired patch, so "
                    "frame lengths cannot be derived from the patch maps."
                )
            src_offsets = [0]
        else:
            if len(frame_lengths) != len(patch_maps):
                raise ValueError(
                    f"frame_lengths has {len(frame_lengths)} entries but "
                    f"{len(patch_maps)} patch maps were given."
                )
            if any(length <= 0 for length in frame_lengths):
                raise ValueError("All frame_lengths must be positive integers.")
            src_offsets = np.cumsum([0] + list(frame_lengths[:-1])).tolist()

        # MASK, OFFSET AND MERGE ALL PATCHES (LOCALS ONLY UNTIL VALIDATED)
        merged_src: list[np.ndarray] = []
        merged_addr: list[np.ndarray] = []
        merged_universe: list[np.ndarray] = []

        for i, patch_map in enumerate(patch_maps):
            if patch_map.dtype.fields is None or not all(
                field in patch_map.dtype.fields
                for field in ("src", "universe", "address")
            ):
                raise ValueError(
                    f"patch_maps[{i}] must be a structured array (patch_dtype) "
                    "with 'src', 'universe' and 'address' fields."
                )

            # MASK INVALID VALUES
            raw_address = patch_map["address"] - 1
            valid_mask = (raw_address >= 0) & (raw_address < DMX_UNIVERSE_SIZE)

            src = patch_map["src"][valid_mask].astype(np.int64)

            # ENSURE THE PATCH FITS INSIDE ITS DECLARED FRAME
            if frame_lengths is not None and len(src) > 0:
                if int(src.max()) >= frame_lengths[i]:
                    raise ValueError(
                        f"patch_maps[{i}] references src index {int(src.max())} "
                        f"but frame_lengths[{i}] is {frame_lengths[i]}: the "
                        "concatenated frame would be too short."
                    )

            # SHIFT SRC INDICES INTO THE CONCATENATED FRAME SPACE
            merged_src.append(src + src_offsets[i])
            merged_addr.append(raw_address[valid_mask])
            merged_universe.append(patch_map["universe"][valid_mask])

        all_src = np.concatenate(merged_src)
        all_addr = np.concatenate(merged_addr)
        patch_universes = np.concatenate(merged_universe)

        # REJECT DUPLICATE (UNIVERSE, ADDRESS) PAIRS: NUMPY ADVANCED INDEXING
        # IS SILENTLY LAST-WRITE-WINS, WHICH WOULD CORRUPT THE ROUTING
        if len(all_addr) > 0:
            keys = patch_universes.astype(np.int64) * DMX_UNIVERSE_SIZE + all_addr
            unique_keys, counts = np.unique(keys, return_counts=True)
            if np.any(counts > 1):
                dup_keys = unique_keys[counts > 1]
                dup_pairs = [
                    (int(k // DMX_UNIVERSE_SIZE), int(k % DMX_UNIVERSE_SIZE) + 1)
                    for k in dup_keys[:5]
                ]
                raise ValueError(
                    f"Patch collision: {len(dup_keys)} duplicate "
                    "(universe, address) pair(s) in the merged routing. "
                    "First duplicates (universe, 1-based address): "
                    f"{dup_pairs}."
                )

        # ROUTING IS VALID: COMMIT IT
        self.has_patch = True
        self._patch_frame_lengths = (
            list(frame_lengths) if frame_lengths is not None else None
        )
        self._patch_src = all_src
        self._patch_addr = all_addr

        # ENSURE ALL UNIVERSES EXIST, IF NOT BUILD THEM
        unique_universes = np.unique(patch_universes)
        for universe in unique_universes:
            if universe not in self.universe_map:
                self.add_universe(universe)

        # PRECOMPUTE MATRIX ROWS PER UNIQUE UNIVERSE
        self._patch_rows = np.array(
            [self.universe_map[universe] for universe in patch_universes]
        )

        # DYNAMICALLY EXPAND PACKET SIZE ONCE ACROSS ALL PATCHES
        # (packet_size is a single global column width, so the largest
        # clamped per-universe requirement equals the clamped global max)
        if len(all_addr) > 0:
            max_addr = int(np.max(all_addr))
            safe_len = clamp_value(max_addr + 1, 2, DMX_UNIVERSE_SIZE, self.make_even)
            if safe_len > self.packet_size:
                self.set_packet_size(safe_len)

    def set_patch(self, patch_map: np.ndarray):
        """
        Register a patch map directly into the client.

        Convenience wrapper for the single-source case of ``set_patches()``:
        filters invalid addresses, automatically expands the matrix to
        accommodate missing universes, pre-computes O(1) routing indices,
        and dynamically grows the packet size based on the highest utilized
        address. No source offset is applied.

        Parameters
        ----------
        patch_map : np.ndarray
            A structured array (patch_dtype) representing the lighting rig.

        Raises
        ------
        ValueError
            If duplicate ``(universe, address)`` pairs exist in the patch.
        """

        self.set_patches([patch_map])

    # --------------------------------
    # DATA
    # --------------------------------
    def set_dmx_value(self, universe: int, data: np.ndarray | list):
        """
        Inject a 1D payload directly into a specific universe's buffer.

        Parameters
        ----------
        universe : int
            The target universe. Will be auto-registered if it doesn't exist.
        data : np.ndarray | list
            A 1D array or list of 8-bit integer values (0 to 255).
        """

        row_idx = self.universe_map.get(universe)
        if row_idx is None:
            row_idx = self.add_universe(universe)

        data_len = len(data)

        if data_len > self.packet_size:
            self.set_packet_size(data_len)

        safe_len = min(data_len, self.packet_size)
        self.buffer[row_idx, :safe_len] = np.clip(data[:safe_len], 0, 255)

    def set_dmx_matrix(self, matrix: np.ndarray):
        """
        Overwrite the client's internal buffer directly with a pre-calculated 2D matrix.

        Parameters
        ----------
        matrix : np.ndarray
            A 2D uint8 matrix of shape (num_universes, length) containing DMX values.
        """

        num_universes, data_len = matrix.shape
        safe_rows = min(num_universes, self.num_universes)

        if data_len > self.packet_size:
            self.set_packet_size(data_len)

        safe_len = min(data_len, self.packet_size)
        self.buffer[:safe_rows, :safe_len] = np.clip(
            matrix[:safe_rows, :safe_len], 0, 255
        )

    def set_patched_dmx_values(self, source_values: np.ndarray):
        """
        Instantly route normalized floats into the internal buffer using the registered patch.

        Requires ``set_patch()`` or ``set_patches()`` to have been called beforehand.
        Flattens the input array, scales values by 255, and leverages C-backed
        advanced indexing for assignment. With a multi-source patch bound via
        ``set_patches()``, pass the concatenation of all source frames in patch
        order.

        Parameters
        ----------
        source_values : np.ndarray
            An array of floats (0.0 to 1.0) representing the current state of the engine.
        """

        if getattr(self, "has_patch", False) is False:
            print("ERROR: Cannot set from values. No patch map registered.")
            return

        # ALLOCATE-ONCE/FAST-MATH: Scale and clip in a single chained operation
        # Note: Depending on engine stability, using pre-allocated `out=` buffers here
        # could yield even more GC savings if source_values size is constant.
        flat_values = (
            np.clip(np.multiply(source_values, 255.0), 0, 255).astype(np.uint8).ravel()
        )

        # MAP INPUT VALUES TO THE UNIVERSE/ADDRESS MATRIX
        self.buffer[self._patch_rows, self._patch_addr] = flat_values[self._patch_src]

    def set_full_buffer(self, value: int):
        """
        Fill the entire internal buffer with a single DMX value.

        Parameters
        ----------
        value : int
            The 8-bit value to fill (clamped to 0 to 255).
        """

        self.buffer.fill(max(0, min(255, value)))

    # --------------------------------
    # RUN FUNCTIONS
    # --------------------------------

    def run_cross_fade(self):
        """Placeholder for future cross-fade functionality. Not yet implemented."""
        pass

    # --------------------------------
    # REPRESENTATION
    # --------------------------------
    def __str__(self):
        state = "********************************\n"
        state += "CLIENT INIT\n"
        state += f"TARGET IP: {self.target_ip}:{self.port}\n"
        state += f"UNIVERSES: {self.universes}\n"
        state += f"PACKET SIZE: {self.packet_size} \n"
        state += f"BUFFERS: {self.buffer.shape}\n"
        if not self.is_simplified:
            state += f"SUBNET: {self.subnet}\n"
            state += f"NET: {self.net}\n"
        state += f"HAS PATCH: {self.has_patch}\n"
        state += "********************************"

        return state

    # --------------------------------
    # RESOURCE MANAGEMENT
    # --------------------------------
    def close(self):
        """Close the UDP socket."""

        self.socket_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
