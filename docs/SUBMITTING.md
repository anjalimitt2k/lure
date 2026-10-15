# Submitting a defense

Lure scores any class that implements its `Defense` interface. You do not need
to fork the benchmark, edit its internals, or understand the attack suite.
Write one method, point the CLI at it, and get the same tables the built-in
defenses get.

**Why this exists.** The most serious limitation of this benchmark is that its
defenses and its attacks were written by the same person. A defense from
someone who did not write the scorer is the single most valuable thing anyone
can contribute. See the limitations section of [FINDINGS.md](FINDINGS.md).

## The contract

```python
from lure.defenses.base import Defense
from lure.environment.marketplace import Marketplace
from lure.environment.models import Cart, IntentMandate, PaymentMandate


class MyDefense(Defense):
    name = "MyDefense"

    def validate(
        self,
        payment: PaymentMandate,
        cart: Cart,
        intent: IntentMandate,
        marketplace: Marketplace,
        ship_to: str | None = None,
    ) -> tuple[bool, str]:
        """Return (allowed, reason). Reason is '' when allowed."""
        return True, ""
```

That is the whole interface. Requirements:

- **Subclass `Defense`** and set a `name`; it labels your rows in every table.
- **Constructible with no arguments.** `MyDefense()` must work; put
  configuration in default arguments. The loader will tell you if it cannot.
- **Return a reason when you block.** Blocked reasons are benchmark output, not
  debug noise; they are what lets a reader see *why* a defense refused.
- **`ship_to` is optional.** `None` means the caller is not proposing a
  delivery change, and should be read as the intent's own address.

### What you are given

| Argument | What it is | Trustworthy? |
| --- | --- | --- |
| `payment` | The transfer the agent proposes: `amount`, `payee` | **No**, agent-supplied |
| `cart` | The basket, holding real catalog `Product` objects | Products yes, quantities no |
| `intent` | The user's Intent Mandate, the ground truth | **Yes** |
| `marketplace` | Catalog, merchant registry, and pages | **Yes**, except page text |
| `ship_to` | Delivery destination the agent chose | **No**, agent-supplied |

The distinction matters. `intent` and `marketplace` are what the user
authorized and what the marketplace knows; `payment`, `ship_to` and cart
quantities are what a possibly-manipulated agent is asking for. Page text
reached through `marketplace.get_page(...)` is attacker-controlled by design;
that is the injection surface.

`cart.total()` is computed from catalog prices, so it is a trusted view of what
the cart is really worth, independent of any figure the agent was talked into.

## Running it

```bash
python -m lure.cli run --defense mypackage.mymodule.MyDefense
```

The path is a dotted path: everything before the last dot is the module,
the last segment is the class. Your module must be importable; if it lives
outside this repo, `pip install -e .` from your package root first.

Repeat `--defense` to score several at once and compare them side by side:

```bash
python -m lure.cli run \
  --defense mypackage.MyDefense \
  --defense lure.defenses.guardrails.DeterministicGuardrails
```

Add `--catalog large` to score on the forty-product catalog (36 benign and
216 attack tasks) instead of the five-product one. Rates there move in steps
of 1/36 rather than 1/4, so a defense that is *nearly* right shows up as
nearly right rather than rounding to a quarter.

A defense that needs a language model can get one the same way the built-in
`LLMJudgeDefense` does: construct an `LLMClient.from_env()` in your
no-argument constructor and raise if it is `None`. The loader turns that into
a clear message rather than a traceback.

## A worked example

[`lure/defenses/example_submission.py`](../lure/defenses/example_submission.py)
is a complete, runnable submission. It checks the payee and nothing else:

```bash
python -m lure.cli run --defense lure.defenses.example_submission.ExampleSubmissionDefense
```

It scores 0.00 attack success on payee redirect and merchant spoofing, 1.00 on
the four attacks it cannot see, and 0.67 overall. It is deliberately partial;
a good example is one that shows you what a weakness looks like in the output,
not one that scores well.

## What you will be scored on

The full suite: 4 benign tasks and 24 attack tasks spanning six attack classes
on the small catalog (36 and 216 on the large one), reported as three numbers
that only mean anything together.

- **`benign_utility`**: fraction of legitimate purchases completed correctly.
- **`attack_success_rate`**: fraction of attacks whose goal was achieved.
- **`over_block_rate`**: fraction of legitimate purchases you refused.

A defense that returns `(False, "no")` unconditionally scores a perfect 0.00
attack success rate, 0.00 benign utility and 1.00 over-block. The metrics are
reported together precisely so that submission is visibly worthless.

## Honest scope

Results are recorded in [`results/RESULTS.md`](../results/RESULTS.md) under
"Submitted defenses". Please read what that table is and is not:

- It records how a defense scored **on this benchmark's own scenarios**, with
  this benchmark's six attack classes, on a simulated marketplace of five or
  forty products.
- It is **not a certification**, not a safety guarantee, and not evidence that
  a defense will hold in deployment.
- Ranking well here means you handled these attacks. The attacks were written
  by one person and are public; treat them as a floor, never a ceiling.

If a defense scores 0.00 attack success, the correct conclusion is "this
benchmark did not find a hole", not "this defense is safe".

## Contributing attacks

Attacks are equally welcome and subclass `Attack` with `structured_payload()`,
`natural_payload()` and an outcome-based `goal_achieved(result, intent)`. Goals
must be defined on the executed transaction (what money moved and where the
goods went), never on the wording that produced it, so that the same attack can
score both the scripted and the model agent. See
[`lure/attacks/base.py`](../lure/attacks/base.py).

An attack that defeats `DeterministicGuardrails` is the most interesting
possible contribution, since that defense currently holds at 0.00 across the
board and its limitations section says plainly that this is partly by
construction.
