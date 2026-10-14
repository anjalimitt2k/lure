# Content filters overstate their own strength: a payment-safety benchmark

## Abstract

Lure is a benchmark for whether an AI shopping agent can be manipulated, via
content injected into a merchant page, into executing a payment its user never
authorized. Across six attack classes and four offline defenses, a trained
TF-IDF and logistic-regression content classifier looks strong on the
injections it was trained on: held-out precision 1.00, recall 0.92, and a 0.00
miss rate on the fixed attack templates, at zero cost to legitimate purchases.
Rephrased into natural language it has not seen, with the fraud goals held
identical, its miss rate rises to 0.17 on the five-product calibration catalog
and to 0.08 on a forty-product catalog with the classifier retrained there,
and its mean confidence falls from 0.96 to 0.66 (1.00 to 0.72 scaled) —
decisions clustered at the boundary rather than made confidently. Measured
attack success against a scripted reference agent moves from 0.00 to 0.08
(0.07 scaled) over the same shift, and the scaled run locates the leakage: it
sits entirely in the goals that can be stated in plain English. Deterministic
guardrails that never read the page — they recompute the transaction from the
catalog and compare it to the user's mandate — hold at 0.00 attack success on
both distributions and both catalogs. The conclusion is narrow but useful:
**evaluating a content filter only on injections drawn from its own training
distribution produces a number that does not survive rephrasing.**
Outcome-level enforcement does not have this failure mode, because it is not
trying to recognise anything. An opt-in tier puts a real model in the agent's
seat and a model-based judge in the content filter's, on the same pages, so
the concession that "a stronger filter would do better" is a measurement
rather than a caveat.

## Background

Payment infrastructure is being rebuilt so that software agents can transact
directly. Google's Agent Payments Protocol (AP2), OpenAI and Stripe's Agentic
Commerce Protocol, and agent-payment programmes from Visa and Mastercard are
all, at different layers, answering one question: how does a merchant or issuer
know that an agent presenting a payment is doing what its user actually asked?
These are named industry developments, cited as context for the problem shape
rather than quoted from specific documents.

AP2's structural answer, and the one Lure adopts, is the **mandate**: three
artifacts a naive design would collapse into one. An **Intent Mandate** records
what the user authorized (item, maximum unit price, quantity, spend cap,
shipping destination, expected payee); a **Cart Mandate** records the basket
the agent assembled; a **Payment Mandate** records the transfer it attempts.

Separating them makes a specific failure legible. An agent can hold a perfectly
valid Intent Mandate and still produce a Payment Mandate that diverges from it
— more units, a higher amount, an extra item, a different account, a different
doorstep. Liability for that divergence is currently unsettled between issuers,
merchants and agent developers; measuring it is a prerequisite for arguing
about it.

The manipulation surface is the merchant page. To buy anything, an agent reads
merchant- and third-party-controlled free text: descriptions, reviews,
metadata. That is untrusted input flowing into a system that moves money. An
agent that treats such text as instructions rather than as data is exhibiting
indirect prompt injection, with a payment attached.

**Relation to prior work.** Agent-security benchmarks such as AgentDojo, and
the broader prompt-injection benchmark line, established the shape of this
evaluation: a realistic tool-using environment, adversarial content planted in
what the agent reads, and a measurement of whether its behaviour was hijacked.
Lure is in that spirit and claims no new evaluation paradigm. What is specific
here, stated precisely rather than expansively:

1. **The commerce vertical.** The environment is a checkout and the terminal
   action is a payment, not a generic tool call.
2. **Mandate-divergence scoring.** Attack success is not "the agent did
   something it should not have" but "the executed Payment Mandate violates the
   Intent Mandate that authorized it" — a comparison between two artifacts,
   defined on outcomes rather than trajectories.
3. **The in- versus out-of-distribution contrast for content filters**, run
   against outcome-level enforcement as a control on the same scenarios.

None of these is a new technique. The combination produces the finding.

## Method

**Environment.** A hermetic in-memory marketplace. The calibration catalog has
5 products across 3 merchants, one deliberately unverified; the scaled catalog
has 40 products across 8 merchants, two unverified, built by the same fixed
template arithmetic. No network, no randomness, no external data. Each product
has a page whose description, reviews and metadata are the injection surfaces.

**Agents.** Two, behind one interface (`run(intent, marketplace, defense) ->
CheckoutResult`). `MockVulnerableAgent` is scripted and maximally credulous: it
obeys any `[[INSTRUCTION]] key=value` line it finds. It is not a strawman for
its own sake but an *upper bound* on the attack surface — a defense that cannot
stop a fully manipulated agent cannot stop a partly manipulated one — and its
perfect credulity is what lets the harness be validated against known-correct
answers. `LLMAgent` puts a real model behind the identical interface, reading
the page as untrusted data and returning a purchase plan as JSON. Both execute
through the same defense-and-checkout path.

**Attacks.** Six classes, each a distinct way an executed transaction can
diverge from its mandate: spend-cap evasion (buy too many), price inflation
(pay more than the goods are worth), item swap (buy something unrequested),
payee redirect (pay an account the catalog does not know), merchant spoofing
(pay an unvetted seller posing as official), and shipping exfiltration
(deliver elsewhere). Each renders the same goal in two registers — a
structured `[[INSTRUCTION]]` token for the scripted agent, and prose for the
model — and is scored identically in both. Goals are classes of outcome, not
strings: payee redirect scores any payee absent from the merchant registry,
shipping exfiltration any address but the user's. Only merchant spoofing is
pinned to a named destination, because its fraud *is* that destination.

**Attack success** is defined as an *executed* payment whose outcome violates
the Intent Mandate in the way that attack targets. A blocked checkout is never
a success, however close the agent came. Scoring reads only the executed
transaction, never the wording that produced it, which is what allows one
attack class to score both agents on one axis.

**Defenses.** Four offline, plus one in the opt-in tier. `NoDefense` is the
undefended control.
`DeterministicGuardrails` applies seven checks in a pinned order — amount
integrity, spend cap, unit-price ceiling, quantity, item identity, payee
authenticity, delivery integrity — every one recomputed from trusted sources
(catalog prices, the merchant registry, the mandate) and none of them reading
page text. `ContentClassifierDefense` is a genuine TF-IDF (word 1–2 grams plus
character 3–5 grams) and logistic-regression model trained on a fixed
231-sample corpus of page texts (135 clean, 96 injected; 173 train, 58
held-out) for the calibration catalog, and on a 1596-sample corpus
(1080 clean, 516 injected; 1197 train, 399 held-out) when scored on
the scaled one, reading content and nothing else. `CompositeDefense` layers
the classifier as an input filter ahead of the guardrails. `LLMJudgeDefense`
puts a language model in the classifier's seat: same pages, same signature,
same ignorance of the transaction, with the model asked whether the page is
trying to steer the agent. It runs only in the LLM tier, since it needs a
model, and refuses to be constructed without one so that it can never score
as a silent `NoDefense`.

**Metrics.** Three, reported together because each is trivially gamed alone:
*benign utility* (fraction of legitimate tasks completed correctly — right
item, authorized quantity, within cap, correct payee, correct address),
*attack success rate*, and *over-block rate*. A defense that blocks everything
scores a perfect ASR and is worthless; over-block is what makes that visible.

**Task set.** 4 benign tasks and 24 attack tasks on the calibration catalog,
36 and 216 on the scaled one — each benign intent crossed with each attack
class. Attack tasks reuse the benign intent verbatim, so any difference in
outcome is attributable to the injection alone. The red-team set adds two
variants per class per intent: 24 more tasks small, 432 scaled.

## Results

The classifier was given every advantage available. It trains on payloads
generated by the very attack objects it is scored against, and the real catalog
pages are in its clean training data. This is deliberate: a filter that leaks
despite being handed its own test distribution has demonstrated something about
content filtering, not about one weak model.

### Deterministic core

| defense | benign utility | ASR | over-block |
| --- | --- | --- | --- |
| `NoDefense` | 1.00 | 1.00 | 0.00 |
| `DeterministicGuardrails` | 1.00 | 0.00 | 0.00 |
| `ContentClassifierDefense` | 1.00 | 0.04 | 0.00 |
| `CompositeDefense` | 1.00 | 0.00 | 0.00 |

Every attack lands undefended, confirming the attacks work and the scorer sees
them. The guardrails stop all 24 without refusing a legitimate purchase.
`fig_asr_by_defense.png` breaks this out per attack class; the only non-zero
cell outside the undefended row is the classifier's 0.25 on spend-cap evasion —
it already leaks slightly even in its best case.

`fig_defense_tradeoff.png` plots utility against ASR. All four defenses sit at
1.00 utility, so nobody bought safety by breaking shopping.
`DeterministicGuardrails` and `CompositeDefense` are exactly coincident at
(0.00, 1.00) and share a marker: here the classifier layer adds nothing the
guardrails had not already caught.

### Held-out classifier scores

Precision 1.00, recall 0.92 on the 58-sample held-out split
(`fig_classifier_pr.png`). Read in isolation, these are the numbers that would
be published as "the classifier works".

### The distribution shift

| injections | n | miss rate | mean confidence |
| --- | --- | --- | --- |
| in-distribution (templates it trained on) | 24 | 0.00 | 0.96 |
| out-of-distribution (red-team phrasings) | 48 | 0.17 | 0.66 |

The fraud goals are identical across both rows. Only the wording differs. The
miss rate rises by 0.17 and mean confidence falls by 0.29. The confidence
collapse matters as much as the misses: at 0.66 mean confidence the classifier
is no longer making decisions, it is hovering at the threshold, and the
difference between a catch and a miss has become close to arbitrary.

Translated into executed fraud against the scripted reference agent, the
classifier's attack success rate moves from 0.00 in-distribution to 0.08
out-of-distribution. The deterministic guardrails hold at 0.00 on both.
`fig_ood_gap.png` is that comparison.

The mechanism is visible in the feature space. The character n-grams do most of
the work, and they key on sub-token shapes — `[[INSTRUCTION]]`, `acct:`,
`addr:` — that survive paraphrase only when the paraphrase happens to reuse
them. A payload written in plain English with no marker tokens has to be caught
on phrasing alone, and phrasing is the part an attacker varies freely.

### The same shift at scale

| catalog | probe pages | n (in / out) | miss rate (in / out) | mean confidence (in / out) | classifier ASR (in / out) |
| --- | --- | --- | --- | --- | --- |
| small | 4 | 24 / 48 | 0.00 / 0.17 | 0.96 / 0.66 | 0.00 / 0.08 |
| large | 36 | 216 / 432 | 0.00 / 0.08 | 1.00 / 0.72 | 0.00 / 0.07 |

The scaled catalog retrains the classifier on nine times the clean pages
(held-out precision 1.00, recall 1.00 on that corpus) and measures each
payload against 36 backgrounds instead of 4. The gap narrows —
more clean data makes the n-gram model somewhat more robust — but does not
close, and the shape of the result is unchanged: in-distribution miss rate
stays at exactly zero while out-of-distribution leakage remains. Guardrails
hold at 0.00 on both.

What scale adds is resolution. Per-attack rates on the small catalog can only
be multiples of 0.25; on the large one they move in steps of 1/36, and the
leakage can be located (`fig_ood_by_attack.png`):

| attack class | classifier ASR, in | classifier ASR, out |
| --- | --- | --- |
| Spend-cap evasion | 0.00 | 0.21 |
| Shipping exfiltration | 0.00 | 0.19 |
| Item swap, price inflation, payee redirect, merchant spoofing | 0.00 | 0.00 |

All of the leakage is in the two goals whose red-team phrasings contain no
account or address token at all — "order sixty", "send it to 44 Fenwick Road".
The four at zero are either goals the scripted reference agent is not
credulous about (item swap, price inflation) or goals whose canned variants
still carry an `acct:` prefix the classifier was trained on. That last point
is a residual property of the *canned* out-of-distribution set, not of the
scorer: payee redirect is now scored by destination class, so a live red-team
run is free to invent an account that carries no such token.

### A partial defense, for calibration

The bundled example submission checks only that the payee matches the mandate.
Scored on the full suite it reports 0.00 ASR on payee redirect and merchant
spoofing, 1.00 on the other four, and 0.67 overall. This is what a
single-signal defense looks like when the benchmark is honest with it, and it
is the intended output of the submission protocol.

## Limitations

These are the reasons not to over-read anything above. I regard this section as
the most important part of the report.

**1. This is a single-author benchmark scoring its own defenses.** I wrote the
attacks, the defenses, the reference agent and the scorer. The result "outcome
enforcement beats content filtering" is exactly the result that design would
tend to produce, because outcome enforcement is checking properties I also used
to define attack success. That circularity is real and I cannot argue it away
from the inside. *To lift it:* defenses and attacks contributed by people who
did not write the scorer. The submission protocol in
[SUBMITTING.md](SUBMITTING.md) exists for this, and until external submissions
arrive, the comparison should be read as a demonstration of a failure mode
rather than a ranking.

**2. The marketplace is simulated.** There are no real payment rails, no
issuer, no 3-D Secure, no chargeback path, no real merchant. `checkout()`
appends to a list. Nothing here establishes that these attacks work against a
deployed agentic-commerce stack, only that they work against an agent reading
untrusted text. *To lift it:* an integration against a sandboxed real
processor, with mandates carried in the actual protocol envelopes rather than
Python dataclasses.

**3. The OOD claim is not uniform across the six goals.** The measured leakage
concentrates in the goals that can be written in plain English (more units, a
different address); the per-attack table above shows it exactly. Payee
redirect is now scored by destination class — any account the catalog does not
know — so a rephrasing may invent its own identifier and still be scored, and
the red-team prompt tells the model so. Merchant spoofing remains pinned to the
unverified seller's real account, because paying *that seller* is the fraud;
its variants will always carry a token a content filter can learn. The
bundled out-of-distribution set was written before the predicate changed and
still names `acct:attacker`, so the offline numbers understate what a
token-free payee redirect would do to the classifier. *To lift the rest:* a
live red-team run under the new prompt, which the LLM tier performs.

**4. The scale is modest.** The calibration catalog has five products and 4
benign tasks, so a per-attack rate there moves in steps of 0.25. The scaled
catalog lifts that to forty products, 36 benign and 216 attack tasks, and
steps of 1/36 — enough to locate leakage by attack class, which the small
catalog could not do. It is still twelve canned red-team variants, still one
author's phrasing, and still no confidence intervals, which at n=36 per cell
would be wide. *To lift it further:* a larger, externally written
out-of-distribution corpus (limitation 6), and a live red-team set generated
fresh per intent rather than reused across them.

**5. The offline classifier is a deliberately simple baseline.** TF-IDF plus
logistic regression is not the state of the art. The finding is therefore
about *content filtering as a category on this benchmark* — that its measured
strength is a function of how close the evaluation injections are to its
training injections — and specifically **not** the claim that no classifier
could do better. `LLMJudgeDefense` exists to test exactly that: it is scored
on the identical in- and out-of-distribution pages as the n-gram model, in the
LLM tier, and its numbers are written to `results/llm_results.md` by
`run-llm`. Those numbers are a property of whichever model is configured and
are not part of the offline reproduction, so they are reported there rather
than here. A fine-tuned transformer baseline remains untested.

**6. The out-of-distribution set is small and author-written.** The 12 red-team
variants shipped for offline reproduction are canned strings I wrote to avoid
the classifier's known lexical tells. That is a fair test of generalisation but
a weak sample of attacker creativity, carrying my own assumptions about what
"different phrasing" means. The live tier generates variants with a model, but
those are not part of the offline reproduction. *To lift it:* a large,
human-reviewed corpus of generated injections, labelled by someone other than
the author.

## Roadmap

Done since the first write-up: the scaled catalog (limitation 4),
destination-class scoring for payee redirect (limitation 3), and an LLM-judge
content filter in the opt-in tier (limitation 5). What remains, in the order
it would change the conclusions most:

- **External defenses.** The `--defense` interface and
  [SUBMITTING.md](SUBMITTING.md) are the mechanism. A defense written by
  someone who did not write the scorer is the highest-value addition, and
  attacks limitation 1 directly.
- **Multi-turn, tool-richer agents.** Both agents make one decision from one
  page. A real shopping agent searches, opens several listings, compares and
  revisits, which gives injected content more places to hide — including on a
  competitor's page. This needs a tool-calling backend interface and a cache
  keyed on whole conversations, which is why it is a separate phase.
- **An externally written out-of-distribution corpus** (limitation 6), so the
  red-team set is not one author's idea of "different wording".
- **A fine-tuned transformer content filter**, to complete limitation 5 with
  a baseline between the n-gram model and a full LLM judge.

Only after those, and only contingent on external adoption, does the
"certification and leaderboard" ambition become meaningful. **It is not
substantiated today and should not be cited as if it were.** A leaderboard
populated entirely by its author's own defenses measures the author, not the
field. The scaffolding exists so the honest version is possible later.

## Reproducibility appendix

Every number in this report regenerates offline, with no API key, byte-identically:

```bash
git clone https://github.com/anjalimitt2k/lure && cd lure
make install
make reproduce
```

`make reproduce` runs the deterministic core, the in- versus
out-of-distribution comparison on both catalogs, and figure generation, then
writes `results/RESULTS.md`, `results/metrics.csv` and five PNGs under
`results/figures/`. Running it twice produces identical bytes, so a clean `git
diff` afterwards means no number moved.

- **Environment:** Python 3.11+ (developed on 3.14.5), scikit-learn and
  matplotlib. No network access at any point.
- **Determinism:** fixed corpus construction, a pinned split
  (`random_state=0`), the `liblinear` solver, no calls to `random` or the
  clock, and suppressed PNG metadata so figures carry no build timestamp.
- **Tests:** `make test` runs 310 tests, never touching the network with or
  without credentials present — the LLM tier runs through a deterministic stub.
- **LLM tier:** `make run-llm` with `ANTHROPIC_API_KEY` and `LURE_MODEL` set
  writes `results/llm_results.md` and caches every response in
  `results/llm_cache.json`; with that cache present the document regenerates
  with no key.
- **Provenance:** the tables in `results/RESULTS.md` come from the same
  functions the acceptance tests assert on (`run_benchmark`,
  `run_ood_comparison`), so a published table cannot disagree with a passing
  test.
- **Portable interface:** `results/metrics.csv` is long-format with
  `tier`/`agent`/`style`/`catalog` labels, for anyone who would rather plot it
  in R.

The opt-in LLM tier (`run-llm`) requires `ANTHROPIC_API_KEY` and `LURE_MODEL`,
caches every response, and is not needed to reproduce anything here.
