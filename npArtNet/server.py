"""High-performance, NumPy-backed implementation of an Art-Net Server."""

import socket
import threading
import numpy as np
from .utils import make_address_mask

DMX_UNIVERSE_SIZE = 512


class ArtnetServer:
    """
    A zero-copy, vectorized Art-Net receiver.

    Uses an $O(1)$ mask dictionary to route incoming UDP packets instantly into
    a 2D NumPy array. Operates on a background daemon thread for non-blocking ingestion.

    Parameters
    ----------
    universes : list[int] | None, optional
        A list of universes to listen for. Default is [0].
    is_simplified : bool, optional
        Whether the incoming data uses simplified 15-bit universe addressing. Default is True.
    subnet : int, optional
        The subnet to listen to (0-15). Default is 0.
    net : int, optional
        The net to listen to (0-127). Default is 0.
    port : int, optional
        The UDP port to bind to. Default is 6454.
    host : str, optional
        The host interface IP to bind to. Default is "" (all interfaces).
    """

    ARTDMX_HEADER = b"Art-Net\x00\x00P\x00\x0e"

    def __init__(
        self,
        universes: list[int] | None = None,
        is_simplified: bool = True,
        subnet: int = 0,
        net: int = 0,
        port: int = 6454,
        host: str = "",
    ):
        self.port = port
        self.host = host
        self.is_simplified = is_simplified
        self.subnet = subnet
        self.net = net

        if universes is None:
            universes = [0]
        self.universes = list(universes)
        self.num_rows = len(self.universes)

        # 1. 2D Matrix to hold all incoming universe data
        self.buffer = np.zeros((self.num_rows, DMX_UNIVERSE_SIZE), dtype=np.uint8)
        self.sequences = np.zeros(self.num_rows, dtype=np.uint8)

        # 2. O(1) Lookup Map: Maps incoming 2-byte network masks to our matrix row indices
        self.mask_to_row: dict[bytes, int] = {}
        self._build_mask_map()

        # 3. Server State & Modern Threading
        self.is_running = False
        self.buffer_lock = threading.Lock()
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._server_thread = threading.Thread(target=self._listen_loop, daemon=True)

    def _build_mask_map(self):
        """Pre-calculates the byte masks for O(1) routing."""
        self.mask_to_row.clear()
        for row_idx, universe in enumerate(self.universes):
            # Convert the bytearray mask to standard immutable bytes for dictionary hashing
            mask = bytes(
                make_address_mask(universe, self.subnet, self.net, self.is_simplified)
            )
            self.mask_to_row[mask] = row_idx

    def start(self):
        """
        Bind the UDP socket and start the background listening thread.
        """

        if self.is_running:
            return

        self.socket_server.bind((self.host, self.port))
        self.is_running = True
        self._server_thread.start()

    def _listen_loop(self):
        """The main high-speed UDP ingestion loop."""
        while self.is_running:
            try:
                # 1. Receive UDP packet
                data, _ = self.socket_server.recvfrom(1024)

                # 2. Validate ArtDmx Header
                if not data.startswith(self.ARTDMX_HEADER):
                    continue

                # 3. O(1) Routing: Look up the 2-byte address mask
                incoming_mask = data[14:16]
                row_idx = self.mask_to_row.get(incoming_mask)

                if row_idx is not None:
                    # 4. Sequence Validation (Drop out-of-order packets)
                    new_seq = data[12]
                    old_seq = self.sequences[row_idx]

                    if (
                        new_seq == 0x00
                        or new_seq > old_seq
                        or (old_seq - new_seq) > 0x80
                    ):
                        self.sequences[row_idx] = new_seq

                        # 5. Extract actual DMX length (bytes 16 & 17 are High Byte First)
                        length = (data[16] << 8) | data[17]
                        safe_length = min(length, DMX_UNIVERSE_SIZE)

                        # 6. ZERO-COPY PARSING: Instantly map the payload to the matrix
                        payload = np.frombuffer(
                            data, dtype=np.uint8, offset=18, count=safe_length
                        )
                        with self.buffer_lock:
                            self.buffer[row_idx, :safe_length] = payload

            except socket.error:
                # Expected when the socket is closed during teardown
                break

    def get_matrix(self) -> np.ndarray:
        """
        Retrieve the current instantaneous state of the network.

        Returns
        -------
        np.ndarray
            A 2D uint8 NumPy array of shape (num_universes, 512) representing
            the latest payloads received for all monitored universes.
        """

        with self.buffer_lock:
            return self.buffer.copy()

    def close(self):
        """
        Gracefully shut down the background listener thread and close the network socket.
        """

        self.is_running = False
        try:
            # Send a dummy packet to ourselves to unblock the recvfrom() call instantly
            dummy_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            dummy_sock.sendto(b"", ("127.0.0.1", self.port))
            dummy_sock.close()
        except Exception:
            pass

        self.socket_server.close()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Listen to Universes 1, 5, and 10
    with ArtnetServer(universes=[1, 5, 10]) as server:
        while server.is_running:
            current_lighting_state = server.get_matrix()
