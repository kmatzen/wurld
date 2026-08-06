"""Static file server with HTTP Range support (python -m http.server lacks it).

    python scripts/range_server.py [port] [root]

Serves single-range GET requests with 206 Partial Content — enough for the
viewer's progressive ranged loading and for wurld.remote over HTTP.
"""

from __future__ import annotations

import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Dev server: never let the browser cache, so ranged loads see fresh bytes.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        try:
            f = open(path, "rb")
        except OSError:
            return super().send_head()
        import os

        size = os.fstat(f.fileno()).st_size
        m = _RANGE.match(self.headers.get("Range", "") or "")
        if not m:
            f.close()
            return super().send_head()
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            f.close()
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        self._range_remaining = end - start + 1
        return _RangeFile(f, end - start + 1)


class _RangeFile:
    def __init__(self, f, remaining):
        self._f, self._remaining = f, remaining

    def read(self, n=-1):
        if self._remaining <= 0:
            return b""
        n = self._remaining if n < 0 else min(n, self._remaining)
        data = self._f.read(n)
        self._remaining -= len(data)
        return data

    def close(self):
        self._f.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8763
    root = sys.argv[2] if len(sys.argv) > 2 else "."
    server = ThreadingHTTPServer(("", port), partial(RangeHandler, directory=root))
    print(f"range server on http://localhost:{port}/ (root: {root})")
    server.serve_forever()


if __name__ == "__main__":
    main()
