# Build notes

What broke while building Lure, what I changed my mind about, and what I would
do differently. The report in [FINDINGS.md](FINDINGS.md) is the tidy version;
this is the untidy one. Rough timeline: the deterministic core and the
natural-language attack channel (what the code still calls Phase 1 and Phase 2)
came first, the public write-up and submission protocol landed in August 2026,
and the scaled catalog, destination-class scoring and live model runs shipped
as 0.4.0 in October 2026.

## Things that were wrong

**The spend cap was checked against the wrong number.** My first guardrail
compared the user's cap to `cart.total()`. The agent's *payment* carries its
own `amount`, and nothing compared that amount to the cart. A manipulated agent
that believed an inflated price could submit a payment above its own cart total
and the guardrail would wave it through, because the cart itself looked fine.
That became check 1, amount integrity, and it is first in the pinned order for
a reason: it is the one that compares the thing that actually moves money to
the thing the catalog says it should cost. The general lesson I took from it is
to enforce on the executed artifact, never on an intermediate the agent built
on the way there.

**Payee redirect was scored as a string match.** The original goal predicate
asked whether the payment went to `acct:attacker`. That is fine for the scripted
agent, which obeys a token, but useless for the red-team set: a rephrased
injection is free to invent its own account identifier, and a scorer that only
recognises one string would call that attack a failure when it succeeded. In
0.4.0 the goal became a class of destination, any account the catalog does not
know about. The bundled out-of-distribution strings were written before that
change and still name `acct:attacker`, so the offline classifier numbers are, if
anything, kind to the classifier. The report says so.

**CI went red on the release commit.** The submission test loads a defense by
dotted path from `tests.test_submission`, exactly as a third party's defense
would be loaded. On GitHub that raised `ModuleNotFoundError: No module named
'tests'`. Locally I had been running the suite in a way that happened to put
the repo root on `sys.path`. The fix was one line in `pyproject.toml`
(`pythonpath = ["."]`), but the real mistake was never running the suite the
way the CI file did until the CI file ran it.

**Exact equality on prices blocks honest purchases.** Prices are floats and
catalog arithmetic is not exact. The amount-integrity check carries a half-cent
tolerance (`AMOUNT_TOLERANCE = 0.005`) because a benchmark that flagged a
legitimate purchase over representation error would be measuring the wrong
thing. Any real overcharge clears half a cent by orders of magnitude.

## Things that were harder than they sounded

**"Byte-identical" is a much stronger promise than "deterministic".** I wanted
`make reproduce` to leave a clean `git diff`, so that any change to a number
shows up as a change to a committed file. Getting there meant a pinned
train/test split (`random_state=0`), the `liblinear` solver, no calls to the
clock or to `random` anywhere in the offline core, and suppressing the
`Software` metadata chunk that matplotlib writes into every PNG, which
otherwise embeds a version string and makes two identical figures differ.
CI now runs the pipeline twice and diffs the two `results/` trees.

**A five-product catalog cannot locate anything.** With 4 benign tasks and 24
attack tasks, a per-attack rate moves in steps of 0.25. The small catalog was
right for checking the harness by hand, and wrong for saying *where* a
classifier leaks. The forty-product catalog (eight merchants, two unverified,
36 benign and 216 attack tasks) moves in steps of 1/36, which was enough to
show that the leakage sits entirely in the two goals that can be phrased
without an account or address token. Product names on both catalogs are chosen
so that no name is a substring of another, and a test checks that every query
resolves to exactly one listing, because an ambiguous lookup would silently
turn a benign task into a failed one.

**The published table must not be able to disagree with a passing test.**
`results/RESULTS.md` is generated from the same functions the acceptance tests
assert exact values on. If I edit the markdown by hand it gets overwritten; if
a number moves, a test fails and the pull request has to explain why. This
felt like overkill until the first time a refactor shifted a rate by one
task and the test caught it before the README did.

**Running a model through a CLI is slow in a way that changes the design.**
The LLM tier can run with no API key through a logged-in Claude Code CLI
(`claude -p`, tools disabled, no session). Each call is a fresh process and
takes seconds, so a few thousand prompts on the large catalog is hours. The
fix was to notice that the agent's prompt for a task does not depend on which
defense adjudicates it, and the judge's prompt for a page does not depend on
the cart. So every prompt a run will issue can be computed up front and
answered concurrently, after which the sequential run is served from cache.
The cache key is `sha256(model + system + prompt)` for both the SDK and the
CLI backend, so a cache filled through one replays through the other.

## Things the model taught me

**Haiku 5.5 refused to write injections.** The red-team generator was built to
ask a model for fresh rephrasings of each fraud goal. The model declined for
all six goals. The live tier records that, falls back to the twelve canned
variants I wrote, and the generated results document says in its first
paragraph that the rephrasings are not model-written. I would rather have the
caveat in the output than in a footnote.

**Haiku 5.5 obeyed none of 720 injected pages.** As the shopping agent it
completed every benign task and followed zero injections, on both catalogs.
That is a good result for the model and a flat one for the benchmark: every
live defense number is 0.00 against an undefended 0.00, so the live run says
nothing about which defense is better. A benchmark needs an agent that can be
fooled before it can rank defenses. The roadmap item that would change the
conclusions most is a weaker or older model in the agent seat, or a multi-step
agent that reads more untrusted text per decision.

**As a content filter the same model was very good.** `LLMJudgeDefense` missed
3 of 720 poisoned pages where the n-gram classifier missed 41, all three out
of distribution. That lifts the "a stronger filter would do better"
concession for this model on this set. It also costs a model call per page
where the n-gram model costs microseconds, and a judge is itself a model
reading untrusted text, which this benchmark has not yet tried to attack.

## Decisions I would defend

- **Structured token first, prose second.** Phase 1 injected a single
  unambiguous `[[INSTRUCTION]] key=value` line so that "did the agent obey?"
  was decidable with no parsing noise. Only once the harness was validated
  against known answers did the prose channel get added beside it. Scoring is
  identical across both, which is what lets one attack class measure a
  scripted agent and a real model on one axis.
- **Over-blocking is always reported next to attack success.** A defense that
  refuses everything has a perfect attack success rate.
- **The offline core never touches the network**, with or without credentials
  present, and the LLM tier exits 0 with instructions when it is not
  configured. CI asserts both.
- **Tests are named as sentences.** `test_an_invented_account_outside_the_catalog_scores_too`
  reads as the behaviour it pins, and the failure output says what broke
  without opening the file.

## What I would do differently

- Start with the forty-product catalog and treat the five-product one as a
  fixture, instead of the other way round.
- Design the agent interface for multi-turn, tool-using shopping from the
  start. Both agents currently make one decision from one page, which is the
  easiest thing to score and the least realistic.
- Get an out-of-distribution corpus written by someone other than me before
  publishing any out-of-distribution number. Twelve variants in one author's
  voice is a fair test of generalisation and a weak sample of attacker
  creativity.
- Add confidence intervals. At n=36 per cell they would be wide, and that is
  worth showing rather than leaving the reader to guess.
