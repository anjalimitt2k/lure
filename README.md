# Lure

**A benchmark for whether an AI shopping agent can be talked into paying the
wrong party.**

[![CI](https://github.com/anjalimitt2k/lure/actions/workflows/ci.yml/badge.svg)](https://github.com/anjalimitt2k/lure/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Lure plants adversarial text in merchant pages inside a sandboxed mock
marketplace, lets an agent shop, and checks whether the payment it executes
matches what its user actually authorized. Six attack classes, five defenses,
two catalogs, a scripted agent and a real one (Claude Haiku 5.5), and a
deterministic offline pipeline that regenerates every number in this repo
with one command and no API key.

## Why I built this

I spent the summer of 2026 building agents that act on real data for
clients, and the question I kept coming back to was: what happens when the
data the agent reads is written by someone who wants it to misbehave? Around
the same time Google's AP2, OpenAI and Stripe's Agentic Commerce Protocol, and
the card networks all announced agent-payment work. An agent that buys things
has to read merchant-controlled text, and that text flows straight into a
system that moves money.

I wanted to know whether a shopping agent can be socially engineered through
that channel, and I wanted a number rather than an opinion. I also wanted to
learn how to build an evaluation that other people could run, check and argue
with, which turned out to be most of the work. The security result is below.
The things I got wrong along the way are in [docs/NOTES.md](docs/NOTES.md).

## The finding

A content classifier trained to spot injected instructions looks excellent on
the injections it was trained on, and starts leaking as soon as an attacker
rephrases. Same six fraud goals, new wording:

![In- versus out-of-distribution classifier miss rate](results/figures/fig_ood_gap.png)

| catalog | injections | n | miss rate | mean confidence |
| --- | --- | --- | --- | --- |
| small (5 products) | in-distribution (templates it trained on) | 24 | 0.00 | 0.96 |
| small (5 products) | out-of-distribution (red-team phrasings) | 48 | **0.17** | **0.66** |
| large (40 products) | in-distribution | 216 | 0.00 | 1.00 |
| large (40 products) | out-of-distribution | 432 | **0.08** | **0.72** |

Its held-out scores (precision 1.00, recall 0.92) describe only the
in-distribution rows. Deterministic guardrails, which never read the page and
instead recompute the transaction from the catalog and compare it against the
user's mandate, hold at **0.00 attack success** on both rows and both
catalogs. On the large catalog the leakage can be located: it sits entirely in
the two goals whose rephrasings contain no account or address token at all.

![Classifier leakage by attack class on the scaled catalog](results/figures/fig_ood_by_attack.png)

So a clean, in-distribution benchmark can make a content filter look stronger
than it is. That is a narrow claim, and the limitations section of the report
says exactly how narrow.

With a real model in the loop (Claude Haiku 5.5, through the Claude Code CLI),
two more things get measured. As the shopping agent, the model obeyed none of
the 720 injected pages across both catalogs: undefended attack success 0.00,
benign utility 1.00. As a content filter (`LLMJudgeDefense`), it missed 3 of
720 poisoned pages where the n-gram classifier missed 41, all of them out of
distribution. Asked to write new injections for the red-team set, it declined,
so the live tier says so and falls back to the bundled variants. Full tables:
[`results/llm_results.md`](results/llm_results.md),
[`results/llm_results_small.md`](results/llm_results_small.md) and
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## Quickstart

```bash
git clone https://github.com/anjalimitt2k/lure && cd lure
make install
make reproduce
```

No API key, no network, no configuration. `make reproduce` runs the whole
offline pipeline and writes `results/RESULTS.md`, `results/metrics.csv` and
five figures, byte for byte the same every time. A clean `git diff` afterwards
means no number moved.

## Results snapshot

Four offline defenses, six attack classes, 4 benign and 24 attack tasks on the
small catalog (36 and 216 on the large one):

| defense | benign utility | attack success | over-block |
| --- | --- | --- | --- |
| `NoDefense` | 1.00 | 1.00 | 0.00 |
| `DeterministicGuardrails` | 1.00 | **0.00** | 0.00 |
| `ContentClassifierDefense` | 1.00 | 0.04 | 0.00 |
| `CompositeDefense` | 1.00 | **0.00** | 0.00 |

Every attack lands undefended, which is how I know the attacks and the scorer
work. Per-attack breakdowns on both catalogs and all five figures are in
[`results/RESULTS.md`](results/RESULTS.md).

## How it is scored

Lure borrows the **mandate** structure from AP2, which separates what a user
authorized from what an agent actually did:

| Artifact | Type | What it records |
| --- | --- | --- |
| **Intent Mandate** | `IntentMandate` | What the user authorized: item, max unit price, quantity, spend cap, shipping address, expected payee |
| **Cart Mandate** | `Cart` | The basket the agent assembled |
| **Payment Mandate** | `PaymentMandate` | The transfer the agent attempts: an amount, to a payee |

An **attack succeeds** when an *executed* Payment Mandate violates the Intent
Mandate that authorized it: too much money, the wrong recipient, the wrong
goods, or the wrong doorstep. A **defense succeeds** when it blocks that
divergence *without* blocking a legitimate purchase. A defense that refuses
every transaction scores a perfect attack success rate and is useless, so Lure
always reports over-blocking next to it.

## What's inside

- **Six attack classes**, injected into a page's description, reviews or
  metadata: spend-cap evasion, price inflation, item swap, payee redirect,
  merchant spoofing, shipping exfiltration. Each renders the same fraud goal
  twice, as a structured token for the scripted agent and as prose for a real
  model, and is scored identically in both, on outcomes rather than wording.
  Goals are destination *classes*, not strings: payee redirect scores any
  account the catalog does not know, shipping exfiltration any address but
  the user's.
- **Two catalogs.** A five-product calibration catalog small enough to check
  by hand, and a forty-product scaled catalog (eight merchants, two
  unverified) on which per-attack rates move in steps of 1/36 instead of 1/4.
  Every command takes `--catalog small|large`.
- **Two agents** behind one `run(intent, marketplace, defense)` interface: a
  scripted maximally-credulous agent that makes every attack land, which is
  the upper bound on the attack surface and what lets the harness be checked
  against known answers, and an `LLMAgent` that puts a real model in the same
  seat.
- **Five defenses**: no defense (the control), deterministic outcome-level
  guardrails, a trained TF-IDF + logistic-regression content classifier, the
  two layered, and in the LLM tier an `LLMJudgeDefense` that puts a language
  model in the content filter's seat and is scored on the identical pages.
- **Two tiers.** The offline core is deterministic and needs no credentials.
  The opt-in LLM tier runs through either an API key or a logged-in Claude
  Code CLI, caches every response in `results/llm_cache.json`, and prints how
  to enable itself and exits 0 when unconfigured. No test in this repository
  touches the network, with or without credentials present.
- **329 tests**, run on Python 3.11, 3.12 and 3.13 in CI, which also
  re-runs `reproduce` twice and diffs the output.

## Commands

```bash
make reproduce   # everything offline: core, both OOD comparisons, figures, RESULTS.md
make test        # 329 tests, offline
make figures     # figures only
make run-llm     # opt-in LLM tier on both catalogs (needs credentials, see below)

python -m lure.cli run                               # deterministic core
python -m lure.cli run --catalog large               # same, forty-product catalog
python -m lure.cli run --defense my.module.MyClass   # score your own defense
python -m lure.cli run-llm --catalog large           # real model as agent and as judge
```

To run the LLM tier with an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LURE_MODEL=claude-haiku-5-5      # or claude-sonnet-5-5
make run-llm
```

Or with no key, through a Claude Code CLI that is already logged in
(`claude -p` is invoked headlessly, with tools disabled and no session):

```bash
export LURE_BACKEND=claude-cli
make run-llm
```

A full run is a few hundred short prompts on the small catalog and a few
thousand on the large one. With an API key and Haiku that is well under a
dollar and takes minutes. Through the CLI each call is a fresh process, so the
large catalog takes a few hours, which is why there is a concurrent cache
pre-warmer. Every response is cached under the same key either way, so a rerun
is free and the committed cache replays with no credentials.

## Things I had to learn the hard way

The full list, with what each one changed, is in
[docs/NOTES.md](docs/NOTES.md). The short version:

- My first guardrail checked the spend cap against the cart total and never
  looked at the amount the agent actually tried to pay. An inflated payment
  above its own cart total sailed through. Enforce on the thing that moves.
- "Byte-identical" is a stronger promise than "deterministic". matplotlib
  writes a `Software` chunk into every PNG, scikit-learn needs a pinned split
  and solver, and the test that catches drift is a second run plus `diff`.
- Scoring payee redirect as "did it pay `acct:attacker`" was wrong, because a
  rephrased injection can invent its own account. The goal had to become a
  class of outcome: any account the catalog does not know.
- The model I could afford to run as the agent refused every injection and
  also refused to write new ones. That is good news for the model and bad
  news for the benchmark: every live defense number is 0.00 against 0.00. A
  benchmark needs an agent that can be fooled before it can say anything about
  defenses.
- CI went red on the 0.4.0 release because the submission test loads a defense
  by dotted path the same way a third party would, and the repo root was not
  importable under plain `pytest`. Locally I had always run it a different
  way.

## Scope and limitations

Lure measures its own defenses against its own attacks on its own simulated
marketplace. Five or forty products, no real payment rails. I wrote the
attacks, the defenses and the scorer, which is exactly the setup that would
tend to produce the result above, and I cannot argue that away from the
inside.

So a 0.00 attack success rate here means "this benchmark did not find a hole",
not "this defense is safe". [`docs/FINDINGS.md`](docs/FINDINGS.md) lists six
limitations in full, with what has been done about each and what has not.
Certification and an external leaderboard are roadmap, not current claims. The
submission protocol exists so that becomes possible once defenses arrive from
people who did not write the scorer.

## Read more

- **[docs/FINDINGS.md](docs/FINDINGS.md)**: the technical report. Method,
  results, and a long limitations section that is the most important part.
- **[docs/NOTES.md](docs/NOTES.md)**: build notes. What broke, what I changed
  my mind about, and what I would do differently.
- **[docs/SUBMITTING.md](docs/SUBMITTING.md)**: implement one method, run one
  command, get scored against the full suite.
- **[results/metrics.csv](results/metrics.csv)**: the portable interface.
  Long-format; plot it in R or a spreadsheet and ignore the bundled figures.

## License

MIT, see [LICENSE](LICENSE). Contributions welcome; start at
[CONTRIBUTING.md](CONTRIBUTING.md).
