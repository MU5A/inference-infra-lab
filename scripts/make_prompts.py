"""Generate the fixed prompt set for Day 1. Deterministic: the same output every run.

Three lengths, because prompt length changes what is being measured: a short prompt isolates
decoding, a long one makes prefill (prompt processing) a visible part of time to first token.
The long text is numbered synthetic statements so no two sentences repeat and nothing is copied
from any source.
"""

import json
import pathlib

ROOT = pathlib.Path("/Users/musa/inference-infra-lab")

SHORT = "Explain what a KV cache is in two sentences."

MEDIUM_PARAGRAPH = (
    "A language model generates text one token at a time. For every new token it must attend to "
    "all earlier tokens, so the keys and values computed for earlier positions are stored and "
    "reused instead of being recomputed. This stored state is called the KV cache. It grows "
    "linearly with the length of the conversation and with the number of layers and attention "
    "heads, and it competes with the model weights for memory. Serving systems therefore spend "
    "much of their effort deciding which cache entries to keep, share, compress or evict. "
)
MEDIUM = (
    MEDIUM_PARAGRAPH
    * 2
    + "Based on the passage above, list three trade-offs a serving engineer faces when sizing "
    "the KV cache."
)


def statement(i: int) -> str:
    units = ["layer", "head", "token", "batch", "request", "block", "slot", "queue"]
    a, b = units[i % len(units)], units[(i * 3 + 1) % len(units)]
    return (
        f"Observation {i}: when the {a} count is {i * 7 % 61 + 3} and the {b} count is "
        f"{i * 11 % 53 + 2}, the measured latency was {i * 13 % 89 + 10} milliseconds."
    )


LONG = (
    "Here is a numbered log of observations from a benchmark.\n"
    + "\n".join(statement(i) for i in range(1, 46))
    + "\nSummarize the log above in one sentence."
)

(ROOT / "prompts" / "day01.json").write_text(
    json.dumps({"short": SHORT, "medium": MEDIUM, "long": LONG}, indent=2) + "\n"
)
print({"short_chars": len(SHORT), "medium_chars": len(MEDIUM), "long_chars": len(LONG)})
