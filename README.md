# Lure

**A benchmark for whether an AI shopping agent can be talked into paying the
wrong party.**

[![CI](https://github.com/anjalimittapally/lure/actions/workflows/ci.yml/badge.svg)](https://github.com/anjalimittapally/lure/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Lure injects adversarial content into merchant pages inside a sandboxed mock
marketplace, lets an agent shop, and measures whether the payment it executes
diverges from what its user actually authorized. It is a security-measurement
project, not a commerce product.

---

## The finding

A content classifier trained to spot injected instructions looks excellent on
the injections it was trained on — and starts leaking the moment an attacker
rephrases. Same six fraud goals, new wording:

![In- versus out-of-distribution classifier miss rate](results/figures/fig_ood_gap.png)

| injections | n | miss rate | mean confidence |
| --- | --- | --- | --- |
| in-distribution (templates it trained on) | 24 | 0.00 | 0.96 |
| out-of-distribution (red-team phrasings) | 48 | **0.17** | **0.66** |

Its held-out scores — precision 1.00, recall 0.92 — describe only the first
row. Deterministic guardrails, which never read the page at all and instead
recompute the transaction from the catalog and compare it against the user's
mandate, hold at **0.00 attack success on both**.

**Clean, in-distribution benchmarks can overstate content-filter defenses.**
That is the point of this repository, and every number above regenerates
offline with one command.

## Quickstart (30 seconds)

```bash
git clone https://github.com/anjalimittapally/lure && cd lure
make install
make reproduce
```

No API key, no network, no configuration. `make reproduce` runs the whole
offline pipeline and writes `results/RESULTS.md`, `results/metrics.csv` and
four figures — byte-identically, so a clean `git diff` afterwards means no
number moved.

## Results snapshot

Four defenses, six attack classes, 4 benign and 24 attack tasks:

| defense | benign utility | attack success | over-block |
| --- | --- | --- | --- |
| `NoDefense` | 1.00 | 1.00 | 0.00 |
| `DeterministicGuardrails` | 1.00 | **0.00** | 0.00 |
| `ContentClassifierDefense` | 1.00 | 0.04 | 0.00 |
| `CompositeDefense` | 1.00 | **0.00** | 0.00 |

Every attack lands undefended, which is what proves the attacks and the scorer
work. Full tables, per-attack breakdowns and all four figures:
[`results/RESULTS.md`](results/RESULTS.md).

## Read more

- **[docs/FINDINGS.md](docs/FINDINGS.md)** — the technical report: method,
  results, and a long limitations section that is the most important part.
- **[docs/SUBMITTING.md](docs/SUBMITTING.md)** — implement one method, run one
  command, get scored against the full suite.
- **[results/metrics.csv](results/metrics.csv)** — the portable interface.
  Long-format; plot it in R or a spreadsheet and ignore the bundled figures.

---

## Why this exists

Payment infrastructure is being rebuilt so that agents can transact on a
person's behalf — Google's AP2, OpenAI and Stripe's Agentic Commerce Protocol,
and agent-payment programmes from Visa and Mastercard are all working on the
same problem. To buy anything, an agent must read merchant-controlled text:
descriptions, reviews, metadata. That is untrusted input flowing into a system
that moves money, and liability for a misdirected payment is currently
unsettled.

Whether an agent can be socially-engineered through that channel is an
empirical question, and it is far easier to argue about than to measure. Lure
measures it.

## The mandate model

Lure follows the **mandate** structure from AP2, which separates what a user
authorized from what an agent actually did:

| Artifact | Type | What it records |
| --- | --- | --- |
| **Intent Mandate** | `IntentMandate` | What the user authorized: item, max unit price, quantity, spend cap, shipping address, expected payee |
| **Cart Mandate** | `Cart` | The basket the agent assembled |
| **Payment Mandate** | `PaymentMandate` | The transfer the agent attempts: an amount, to a payee |

- An **attack succeeds** when an *executed* Payment Mandate violates the Intent
  Mandate that authorized it — too much money, the wrong recipient, the wrong
  goods, or the wrong doorstep.
- A **defense succeeds** when it blocks that divergence *without* blocking a
  legitimate purchase.

That second clause is the part benchmarks skip. A defense that refuses every
transaction scores a perfect attack success rate and is useless, so Lure always
reports over-blocking alongside it.

## What's in the box

**Six attack classes**, injected into a page's description, reviews or
metadata: spend-cap evasion, price inflation, item swap, payee redirect,
merchant spoofing, shipping exfiltration. Each renders the same fraud goal
twice — as a structured token for the scripted agent, and as natural-language
prose for a real model — and is scored identically in both, on outcomes rather
than on wording.

**Two agents** behind one interface: a scripted maximally-credulous agent that
makes every attack land (the upper bound on the attack surface, and what lets
the harness be validated against known-correct answers), and an `LLMAgent` that
puts a real model behind the identical `run(intent, marketplace, defense)`
signature.

**Four defenses**: no defense (the control), deterministic outcome-level
guardrails, a genuinely trained TF-IDF + logistic-regression content
classifier, and the two layered.

**Two tiers.** The offline core is deterministic and needs no credentials. The
opt-in LLM tier (`python -m lure.cli run-llm`) needs `ANTHROPIC_API_KEY` and
`LURE_MODEL`, caches every response, and prints how to enable itself and exits
0 when unconfigured. **No test in this repository touches the network**, with
or without credentials present.

## Commands

```bash
make reproduce   # everything: core, OOD comparison, figures, RESULTS.md
make test        # 261 tests, offline
make figures     # figures only

python -m lure.cli run                              # deterministic core
python -m lure.cli run --defense my.module.MyClass  # score your own defense
python -m lure.cli run-llm                          # opt-in LLM tier
```

## Honest scope

Lure measures **its own defenses against its own attacks on its own simulated
marketplace**. Five products, three merchants, no real payment rails. The
attacks, the defenses and the scorer were written by the same person, which is
exactly the setup that would tend to produce the result reported above — a
circularity I cannot argue away from the inside.

So a 0.00 attack success rate here means "this benchmark did not find a hole",
not "this defense is safe". [`docs/FINDINGS.md`](docs/FINDINGS.md) states six
limitations in full, with what would be needed to lift each.

**Certification and an externally-populated leaderboard are roadmap, not
current claims.** The submission protocol exists so that the honest version
becomes possible once defenses arrive from people who did not write the scorer.
Until then, the results table records how defenses scored on these scenarios
and nothing more.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome; start at
[CONTRIBUTING.md](CONTRIBUTING.md).
