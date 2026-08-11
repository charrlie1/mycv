"""
mycv.streaming
===============
Optional network video-stream ingestion (RTSP / HTTP / HTTPS / UDP / RTP /
local files / webcams) via PyAV, a Python wrapper around FFmpeg.

This module is entirely optional and outside the pure-NumPy mathematical
core of `mycv` — it exists to support live deployment. It requires the
optional `av` package:

    pip install av

If `av` is not installed, importing this module still succeeds (so
`from mycv import *` never breaks), but instantiating `StreamReader`
raises a clear ImportError.

Classes
-------
StreamReader : Background-thread stream reader with automatic
               reconnection and latest-frame-only queueing.

Why a background thread + a single-slot queue?
------------------------------------------------
Network streams fail (dropped connections, camera reboots, Wi-Fi
hiccups) — a production reader needs to reconnect rather than crash.
And for live processing, an old undelivered frame is worse than useless
(it adds latency); a decode thread should always be able to write the
newest frame, discarding whatever stale frame the consumer hasn't
picked up yet. `queue.Queue(maxsize=1)` plus a non-blocking `get_nowait`
drop-then-put gives exactly that: the consumer always reads the most
recent frame available, with reconnection running transparently in the
background.
"""

import queue
import threading
import time

try:
    import av
    _HAS_AV = True
except ImportError:                      # pragma: no cover - optional dep
    av = None
    _HAS_AV = False


class StreamReader:
    """
    Background video-stream reader with reconnection and frame-dropping.

    Parameters
    ----------
    url             : str  stream URL — rtsp://, http(s)://, udp://, rtp://,
                      a local file path, or an FFmpeg device string
    rtsp_transport  : str  'tcp' (reliable, higher latency) or 'udp'
                      (lower latency, may drop packets). Only relevant
                      for rtsp:// URLs.
    timeout_us      : int  FFmpeg socket timeout in microseconds
    reconnect_delay : float  seconds to wait before retrying after a
                      stream error

    Usage
    -----
        reader = StreamReader("rtsp://192.168.1.100:554/stream").start()
        frame, timestamp = reader.read(timeout=1.0)
        if frame is not None:
            ...
        reader.stop()
    """

    def __init__(
        self,
        url: str,
        rtsp_transport: str = "tcp",
        timeout_us: int = 5_000_000,
        reconnect_delay: float = 2.0,
    ) -> None:
        if not _HAS_AV:
            raise ImportError(
                "mycv.streaming.StreamReader requires the optional 'av' "
                "package (PyAV, an FFmpeg wrapper). Install it with:\n"
                "    pip install av"
            )
        self.url = url
        self.options = {
            "rtsp_transport": rtsp_transport,
            "stimeout": str(timeout_us),
        }
        self.reconnect_delay = reconnect_delay

        self._queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread = None

    def start(self) -> "StreamReader":
        """Start the background decode/reconnect thread. Returns self."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while self._running:
            container = None
            try:
                container = av.open(self.url, options=self.options)
                for frame in container.decode(video=0):
                    if not self._running:
                        break
                    arr = frame.to_ndarray(format="rgb24")
                    timestamp = time.perf_counter()
                    self._push(arr, timestamp)
            except Exception:
                # Any decode/connection failure: back off and retry.
                # Network streams fail; the caller should not have to
                # handle this — reconnection happens transparently.
                pass
            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass
            if self._running:
                time.sleep(self.reconnect_delay)

    def _push(self, frame, timestamp: float) -> None:
        """Drop any stale undelivered frame, then push the newest one."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait((frame, timestamp))
        except queue.Full:
            pass  # lost a race with another producer step; drop this frame

    def read(self, timeout: float = 1.0):
        """
        Return the most recent (frame, timestamp) pair, waiting up to
        `timeout` seconds for one to arrive if the queue is empty.

        Returns
        -------
        (frame, timestamp) : frame is an (H, W, 3) uint8 RGB array and
            timestamp is a `time.perf_counter()` float, or (None, None)
            if no frame arrived within `timeout` seconds.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stop(self) -> None:
        """Stop the background thread and release the stream."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.reconnect_delay + 1.0)
