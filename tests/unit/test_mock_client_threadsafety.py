"""Thread-safety contract for :class:`MockLLMClient` under Stage 7's runner.

The runner calls the same mock from multiple worker threads. The queue
pops and the call log appends must stay consistent under that load.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel

from src.llm.base import LLMRequest, LLMUsage
from src.llm.mock_client import MockLLMClient


class _Echo(BaseModel):
    value: int


def _make_request(value: int) -> LLMRequest:
    return LLMRequest(
        system_prompt="sys",
        user_prompt=f"ask {value}",
        temperature=0.0,
        max_output_tokens=64,
    )


class TestMockClientThreadSafety:
    def test_scripted_responses_are_popped_exactly_once(self) -> None:
        n = 200
        items = [_Echo(value=i) for i in range(n)]
        client = MockLLMClient(
            model_name="mock",
            structured_script=items,
            usage=LLMUsage(input_tokens=1, output_tokens=1),
        )

        received: list[int] = []
        received_lock = threading.Lock()

        def worker(i: int) -> None:
            resp = client.generate_structured(_make_request(i), _Echo)
            with received_lock:
                received.append(resp.parsed.value)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(received) == list(range(n))
        assert client.structured_script_remaining == 0
        # Every call should be logged exactly once.
        assert len(client.calls) == n

    def test_text_script_is_threadsafe(self) -> None:
        n = 100
        client = MockLLMClient(
            model_name="mock",
            text_script=[f"answer-{i}" for i in range(n)],
        )

        results: list[str] = []
        results_lock = threading.Lock()

        def worker(i: int) -> None:
            out = client.generate_text(_make_request(i))
            with results_lock:
                results.append(out.text)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == sorted(f"answer-{i}" for i in range(n))
        assert client.text_script_remaining == 0

    def test_callable_mode_under_threads_does_not_deadlock(self) -> None:
        """``text_fn`` runs outside the lock so slow fixtures don't stall."""
        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def fn(_req: LLMRequest) -> str:
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            # Simulate slow "model" without holding the mock's lock.
            import time

            time.sleep(0.02)
            with lock:
                in_flight -= 1
            return "ok"

        client = MockLLMClient(model_name="mock", text_fn=fn)

        threads = [
            threading.Thread(target=lambda i=i: client.generate_text(_make_request(i)))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak > 1, f"text_fn ran serially (peak={peak}); lock contention suspected."
