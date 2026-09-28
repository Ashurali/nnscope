"""A tiny live server (stdlib only): the dashboard page, the run history, and a Server-Sent-Events stream."""
from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .export import render_page

PUSH_INTERVAL = 0.1      # seconds: batch messages → at most ~10 pushes per second per browser
HEARTBEAT = 15.0


def _handler_for(scope):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # keep the user's console clean
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(render_page(None, live=True).encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/history":
                self._send(json.dumps(scope.data()).encode("utf-8"), "application/json")
            elif path == "/api/stream":
                self._stream()
            else:
                self._send(b"not found", "text/plain", 404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q: queue.Queue = queue.Queue()
            with scope._lock:
                scope._listeners.append(q)
            last = time.time()
            try:
                while True:
                    batch = []
                    try:
                        batch.append(q.get(timeout=1.0))
                        while True:
                            batch.append(q.get_nowait())
                    except queue.Empty:
                        pass
                    if batch:
                        self.wfile.write(f"data: {json.dumps(batch)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last = time.time()
                        if any(m.get("type") == "end" for m in batch):
                            break
                        time.sleep(PUSH_INTERVAL)
                    elif time.time() - last > HEARTBEAT:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last = time.time()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                with scope._lock:
                    if q in scope._listeners:
                        scope._listeners.remove(q)

    return Handler


def start_server(scope, host: str = "127.0.0.1", port: int = 8765):
    """Start serving in a daemon thread on the first free port ≥ `port` (0 = let the OS choose)."""
    handler = _handler_for(scope)
    last_err = None
    for p in ([0] if port == 0 else list(range(port, port + 20)) + [0]):
        try:
            server = ThreadingHTTPServer((host, p), handler)
            break
        except OSError as err:
            last_err = err
    else:
        raise OSError(f"nnscope: no free port found ({last_err})")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="nnscope-server").start()
    return server, f"http://{host}:{server.server_address[1]}/"
