"""A streaming client for OpenAI-compatible chat endpoints that measures latency from the outside.

Definitions, so every number means one thing:

- **TTFT** (time to first token): from just before the request is sent to the arrival of the
  first stream event that carries visible text. It includes the network round trip, request
  parsing and prompt processing on the server. A reused connection is assumed.
- **ITL** (inter-token latency): the gap between the arrivals of consecutive text-bearing events.
  A server may pack more than one token into an event (for example while it holds back an
  incomplete UTF-8 character), so ITL is per event and can slightly understate per-token gaps.
- **Decode rate**: (text events after the first) divided by the time from the first to the last
  text event. This counts events, not the server's `usage.completion_tokens`, because that count
  includes the end-of-turn token, which carries no text, and would overstate the rate. The
  usage-based rate is kept beside it as `decode_tps_usage`, and `unseen_tokens` (tokens the
  server counted minus text events seen) shows whether it packs several tokens into one event.
  Zero or one is normal. One is the end token. A larger value means events are carrying more than
  one token each, and the event-based rate then understates the true token rate.
- **Cached tokens**: the number of prompt tokens the server reused from an earlier request. A
  nonzero value means prompt processing was skipped, so TTFT is not a cold measurement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

CHAT_PATH = "/v1/chat/completions"


@dataclass
class StreamResult:
    start_ns: int
    end_ns: int
    text_arrivals_ns: list[int] = field(default_factory=list)
    text: str = ""
    usage: dict | None = None
    server_timings: dict | None = None
    finish_reason: str | None = None
    malformed_events: int = 0
    error: str | None = None
    events: list[tuple[int, dict]] | None = None  # only kept on request, for harness calibration

    @property
    def n_tokens(self) -> int:
        if self.usage and isinstance(self.usage.get("completion_tokens"), int):
            return self.usage["completion_tokens"]
        return len(self.text_arrivals_ns)

    @property
    def tokens_estimated(self) -> bool:
        return not (self.usage and isinstance(self.usage.get("completion_tokens"), int))


def parse_data_line(line: str) -> tuple[str, dict | None]:
    """Classify one line of a server-sent-events stream.

    Returns ("skip", None) for blank lines, comments and non-data fields, ("done", None) for the
    end marker, ("bad", None) for a data line that is not valid JSON, and ("event", dict) for a
    parsed JSON object.
    """
    line = line.rstrip("\r\n")
    if not line.startswith("data:"):
        return "skip", None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "done", None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return "bad", None
    if not isinstance(event, dict):
        return "bad", None
    return "event", event


def _delta_text(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if isinstance(content, str) else ""


def stream_chat(
    client: httpx.Client, base_url: str, payload: dict, *, keep_events: bool = False
) -> StreamResult:
    """Send one streaming chat request and record when each piece of text arrives.

    Never raises for network or HTTP problems: they are returned in `error`, with whatever
    arrived before the failure.
    """
    body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    start = time.perf_counter_ns()
    result = StreamResult(start_ns=start, end_ns=start, events=[] if keep_events else None)
    parts: list[str] = []
    try:
        with client.stream("POST", base_url.rstrip("/") + CHAT_PATH, json=body) as response:
            if response.status_code != 200:
                result.error = f"http {response.status_code}"
                return result
            for line in response.iter_lines():
                arrived = time.perf_counter_ns()
                kind, event = parse_data_line(line)
                if kind == "done":
                    break
                if kind == "bad":
                    result.malformed_events += 1
                    continue
                if kind != "event" or event is None:
                    continue
                if result.events is not None:
                    result.events.append((arrived, event))
                text = _delta_text(event)
                if text:
                    result.text_arrivals_ns.append(arrived)
                    parts.append(text)
                if isinstance(event.get("usage"), dict):
                    result.usage = event["usage"]
                if isinstance(event.get("timings"), dict):
                    result.server_timings = event["timings"]
                choices = event.get("choices") or []
                if choices and isinstance(choices[0], dict) and choices[0].get("finish_reason"):
                    result.finish_reason = choices[0]["finish_reason"]
    except httpx.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.end_ns = time.perf_counter_ns()
        result.text = "".join(parts)
    return result


def metrics(result: StreamResult) -> dict:
    """Latency numbers for one request, in seconds and tokens per second."""
    arrivals = result.text_arrivals_ns
    ttft = (arrivals[0] - result.start_ns) / 1e9 if arrivals else None
    gaps = [(b - a) / 1e9 for a, b in zip(arrivals, arrivals[1:], strict=False)]
    decode_tps = decode_tps_usage = None
    if len(arrivals) >= 2:
        span = (arrivals[-1] - arrivals[0]) / 1e9
        if span > 0:
            decode_tps = (len(arrivals) - 1) / span
            if not result.tokens_estimated and result.n_tokens >= 2:
                decode_tps_usage = (result.n_tokens - 1) / span
    details = (result.usage or {}).get("prompt_tokens_details") or {}
    return {
        "ttft_s": ttft,
        "itl_s": gaps,
        "decode_tps": decode_tps,
        "decode_tps_usage": decode_tps_usage,
        "unseen_tokens": (None if result.tokens_estimated else result.n_tokens - len(arrivals)),
        "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
        "server_predicted_tps": (result.server_timings or {}).get("predicted_per_second"),
        "server_prompt_tps": (result.server_timings or {}).get("prompt_per_second"),
        "total_s": (result.end_ns - result.start_ns) / 1e9,
        "completion_tokens": result.n_tokens,
        "prompt_tokens": (result.usage or {}).get("prompt_tokens"),
        "tokens_estimated": result.tokens_estimated,
        "text_events": len(arrivals),
        "malformed_events": result.malformed_events,
        "finish_reason": result.finish_reason,
        "error": result.error,
    }
