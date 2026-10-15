# Contributing

## Submitting a defense or an attack

That is the main thing this project wants. The full contract, a worked example
and the scope of the results table are in
**[docs/SUBMITTING.md](docs/SUBMITTING.md)**.

## Development setup

```bash
make install     # venv + dependencies + editable install
make test        # 329 tests, offline, never touches the network
make reproduce   # regenerate every artifact under results/
```

## Ground rules

These are the invariants that keep the benchmark worth citing. A change that
breaks one of them needs a very good argument.

**The offline core stays offline.** No network calls, no API keys, no
randomness, no clock reads in `lure/` outside the opt-in LLM tier
(`lure/llm/`, `lure/defenses/llm_judge.py`, `lure/eval/llm_results.py`,
`lure/eval/warm.py`). The
test suite must pass with credentials absent *and* present; every test drives
the LLM tier through a deterministic stub.

**Both catalogs stay deterministic.** The five-product calibration catalog and
the forty-product scaled one are built by fixed template arithmetic. A new
product name must not be a substring of another; `tests/test_catalog.py`
checks that every query resolves to exactly one listing.

**Results stay reproducible.** `make reproduce` must produce byte-identical
artifacts on repeated runs. If you add a figure or a table, it must regenerate
from live results rather than hardcoded values, and carry no timestamp.

**`results/RESULTS.md` is generated, never edited.** It is written from the
same functions the acceptance tests assert on, so that a published table cannot
disagree with a passing test. Editing it by hand defeats the point and will be
overwritten.

**Acceptance numbers are pinned deliberately.** `tests/test_acceptance.py`
asserts exact values. If your change moves one, that is not automatically a
failure, but the pull request needs to explain *why* the number moved, in the
same way the findings write-up would have to.

**Attack goals are outcome-based.** Scored on what the executed transaction
did, never on the wording that produced it. This is what lets one attack class
score both the scripted and the model agent on one axis.

**Limitations get documented, not hidden.** If you find a way the benchmark
flatters itself, the fix is to write it down in
[docs/FINDINGS.md](docs/FINDINGS.md), whether or not you can also fix the code.

## Style

Small typed functions with docstrings that say *why*, matching the surrounding
code. Tests named as sentences describing the behaviour being pinned.
