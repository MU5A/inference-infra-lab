"""A fake OpenAI-compatible streaming server with exactly controllable timing.

It copies the event shapes a real llama.cpp server sends (a first event whose content is null,
one text event per token, a finish event, a final event with usage and timings, then [DONE]) and
stamps every event with the moment it was written (`x_sent_ns`). That lets tests check the
client's arithmetic against known delays, and lets the overhead script measure how far the
client's clock readings drift from the truth.

Run it alone with:  python -m inferlab.fakeserver --first-delay 0.1 --interval 0.02
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Behavior:
    first_delay_s: float = 0.05  # request received to first text event
    interval_s: float = 0.01  # gap between text events
    n_tokens: int = 8
    prompt_tokens: int = 20
    cached_tokens: int = 0
    mode: str = "ok"  # ok, http500, drop_mid, malformed, no_tokens, noisy, packed, no_usage
    tokens_per_event: int = 1


def _wait_until(deadline_ns: int) -> None:
    remaining = (deadline_ns - time.perf_counter_ns()) / 1e9
    if remaining > 0:
        time.sleep(remaining)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # keep test output quiet
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass  # the client closed an idle or finished connection

    @property
    def behavior(self) -> Behavior:
        return self.server.behavior  # type: ignore[attr-defined]

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _event(self, payload: dict | str, *, raw: bool = False, stamp: bool = True) -> None:
        if isinstance(payload, dict):
            if stamp:
                payload = {**payload, "x_sent_ns": time.perf_counter_ns()}
            text = json.dumps(payload)
        else:
            text = payload
        line = text if raw else f"data: {text}"
        self._chunk((line + "\n\n").encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)
        try:
            self.server.last_request = json.loads(raw_body)  # type: ignore[attr-defined]
        except json.JSONDecodeError:
            self.server.last_request = None  # type: ignore[attr-defined]
        b = self.behavior
        received = time.perf_counter_ns()
        if b.mode == "http500":
            body = b'{"error":"boom"}'
            self.send_response(500)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def text_event(i: int) -> dict:
            text = "".join(f"t{i}_{j} " for j in range(b.tokens_per_event))
            return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}

        self._event({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]})
        events = b.n_tokens if b.mode != "no_tokens" else 0
        for i in range(events):
            _wait_until(received + int(b.first_delay_s * 1e9) + int(i * b.interval_s * 1e9))
            if b.mode == "noisy":
                self._chunk(b": keep-alive\n\n")
                self._chunk(b"event: message\r\n")
            if b.mode == "malformed" and i == 1:
                self._event("data: {this is not json", raw=True)
            self._event(text_event(i))
            if b.mode == "drop_mid" and i == 2:
                self.close_connection = True
                return  # no terminating chunk: the client sees an incomplete body

        self._event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        completion = events * b.tokens_per_event + (1 if events else 0)  # the end token
        final: dict = {"choices": []}
        if b.mode != "no_usage":
            final["usage"] = {
                "completion_tokens": completion,
                "prompt_tokens": b.prompt_tokens,
                "total_tokens": completion + b.prompt_tokens,
                "prompt_tokens_details": {"cached_tokens": b.cached_tokens},
            }
            final["timings"] = {
                "prompt_n": b.prompt_tokens - b.cached_tokens,
                "predicted_n": completion,
                "predicted_per_second": (1 / b.interval_s) if b.interval_s else None,
            }
        self._event(final)
        self._event("data: [DONE]", raw=True, stamp=False)
        self._chunk(b"")  # the terminating zero-length chunk


class FakeServer:
    """Start with `with FakeServer(Behavior(...)) as server:`. `server.url` is the base URL."""

    def __init__(self, behavior: Behavior | None = None, port: int = 0):
        self.behavior = behavior or Behavior()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._httpd.daemon_threads = True
        self._httpd.behavior = self.behavior  # type: ignore[attr-defined]
        self._httpd.last_request = None  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def last_request(self) -> dict | None:
        return self._httpd.last_request  # type: ignore[attr-defined]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> FakeServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fake streaming server.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--first-delay", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=0.01)
    parser.add_argument("--tokens", type=int, default=8)
    args = parser.parse_args()
    behavior = Behavior(args.first_delay, args.interval, args.tokens)
    with FakeServer(behavior, args.port) as server:
        print(server.url, flush=True)
        threading.Event().wait()


if __name__ == "__main__":
    main()
