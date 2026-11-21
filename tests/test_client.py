"""The streaming client against a fake server whose timing and faults are known exactly."""

from __future__ import annotations

import socket

import httpx
import pytest

from inferlab.client import metrics, parse_data_line, stream_chat
from inferlab.fakeserver import Behavior, FakeServer


def run(behavior: Behavior, **kwargs):
    with FakeServer(behavior) as server, httpx.Client(timeout=10) as client:
        result = stream_chat(client, server.url, {"messages": []}, **kwargs)
        return result, metrics(result), server.last_request


class TestParsing:
    def test_blank_lines_comments_and_other_fields_are_skipped(self):
        for line in ["", "\n", ": keep-alive", "event: message", "id: 4", "retry: 100", "garbage"]:
            assert parse_data_line(line) == ("skip", None), line

    def test_done_marker(self):
        assert parse_data_line("data: [DONE]") == ("done", None)
        assert parse_data_line("data:[DONE]\r\n") == ("done", None)

    def test_json_with_and_without_a_space_and_with_crlf(self):
        assert parse_data_line('data: {"a": 1}') == ("event", {"a": 1})
        assert parse_data_line('data:{"a": 1}') == ("event", {"a": 1})
        assert parse_data_line('data: {"a": 1}\r\n') == ("event", {"a": 1})

    @pytest.mark.parametrize(
        "line", ["data: {not json", "data: ", "data: 5", 'data: "text"', "data: [1]"]
    )
    def test_data_that_is_not_a_json_object_is_bad(self, line):
        assert parse_data_line(line) == ("bad", None)


class TestTiming:
    def test_ttft_is_the_first_text_event_not_the_first_event(self):
        """The role event arrives at once. Only the delayed text event counts as the first token."""
        _, m, _ = run(Behavior(first_delay_s=0.15, interval_s=0.02, n_tokens=5))
        assert m["ttft_s"] >= 0.15
        assert m["ttft_s"] < 0.15 + 0.4

    def test_inter_token_gaps_follow_the_interval(self):
        _, m, _ = run(Behavior(first_delay_s=0.02, interval_s=0.03, n_tokens=8))
        assert len(m["itl_s"]) == 7
        assert all(gap >= 0 for gap in m["itl_s"])
        assert 0.02 < sorted(m["itl_s"])[3] < 0.08  # the median gap

    def test_decode_rate_is_events_after_the_first_over_the_time_they_took(self):
        result, m, _ = run(Behavior(first_delay_s=0.02, interval_s=0.02, n_tokens=11))
        span = (result.text_arrivals_ns[-1] - result.text_arrivals_ns[0]) / 1e9
        assert m["decode_tps"] == pytest.approx(10 / span)
        assert 0.1 / 0.02 < m["decode_tps"] <= 1.05 / 0.02  # never faster than the server paces it

    def test_the_usage_based_rate_counts_the_end_token_the_event_rate_does_not(self):
        _, m, _ = run(Behavior(interval_s=0.02, n_tokens=8))
        assert m["completion_tokens"] == 9  # 8 text events plus the end token
        assert m["text_events"] == 8
        assert m["unseen_tokens"] == 1
        assert m["decode_tps_usage"] / m["decode_tps"] == pytest.approx(8 / 7)

    def test_packed_events_are_visible_as_unseen_tokens(self):
        _, m, _ = run(Behavior(interval_s=0.01, n_tokens=5, tokens_per_event=3))
        assert m["text_events"] == 5
        assert m["unseen_tokens"] == 11  # 15 text tokens plus the end token, minus 5 events

    def test_without_usage_tokens_are_the_text_events_and_flagged_as_estimated(self):
        _, m, _ = run(Behavior(n_tokens=6, mode="no_usage"))
        assert m["tokens_estimated"] is True
        assert m["completion_tokens"] == 6
        assert m["decode_tps_usage"] is None and m["unseen_tokens"] is None
        assert m["prompt_tokens"] is None

    def test_cached_prompt_tokens_and_server_timings_are_reported(self):
        _, m, _ = run(Behavior(prompt_tokens=40, cached_tokens=15, interval_s=0.02, n_tokens=4))
        assert m["prompt_tokens"] == 40
        assert m["cached_tokens"] == 15
        assert m["server_predicted_tps"] == pytest.approx(50.0)

    def test_the_request_asks_for_a_stream_with_usage(self):
        _, _, request = run(Behavior(n_tokens=2))
        assert request["stream"] is True
        assert request["stream_options"] == {"include_usage": True}


class TestFaults:
    def test_a_noisy_stream_with_comments_and_event_lines_parses_normally(self):
        _, m, _ = run(Behavior(n_tokens=6, mode="noisy"))
        assert (m["text_events"], m["malformed_events"], m["error"]) == (6, 0, None)

    def test_a_malformed_event_is_counted_and_skipped(self):
        _, m, _ = run(Behavior(n_tokens=6, mode="malformed"))
        assert (m["text_events"], m["malformed_events"], m["error"]) == (6, 1, None)

    def test_an_http_error_is_returned_not_raised(self):
        _, m, _ = run(Behavior(mode="http500"))
        assert m["error"] == "http 500"
        assert m["ttft_s"] is None and m["decode_tps"] is None

    def test_a_dropped_connection_keeps_what_arrived_and_records_the_error(self):
        result, m, _ = run(Behavior(n_tokens=8, mode="drop_mid", interval_s=0.005))
        assert m["error"] is not None
        assert m["text_events"] == 3
        assert result.text.startswith("t0_0 t1_0 t2_0")
        assert m["ttft_s"] is not None

    def test_a_refused_connection_is_returned_not_raised(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with httpx.Client(timeout=2) as client:
            result = stream_chat(client, f"http://127.0.0.1:{port}", {"messages": []})
        assert result.error is not None and result.text == ""

    def test_a_reply_with_no_text_has_no_latency_numbers(self):
        _, m, _ = run(Behavior(mode="no_tokens"))
        assert m["error"] is None
        assert m["ttft_s"] is None and m["decode_tps"] is None and m["itl_s"] == []
        assert m["completion_tokens"] == 0

    def test_a_single_token_reply_has_a_ttft_but_no_decode_rate(self):
        _, m, _ = run(Behavior(n_tokens=1))
        assert m["ttft_s"] is not None
        assert m["itl_s"] == [] and m["decode_tps"] is None


class TestClockSanity:
    def test_the_client_never_sees_an_event_before_the_server_sent_it(self):
        """Both clocks are the same monotonic clock, so arrival minus send must not be negative."""
        result, _, _ = run(Behavior(n_tokens=10, interval_s=0.005), keep_events=True)
        assert result.events
        for arrived, event in result.events:
            assert arrived - event["x_sent_ns"] >= 0
