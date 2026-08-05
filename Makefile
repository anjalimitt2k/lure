# Thin wrappers over the CLI. Everything here runs offline with no API key.
#
# `make reproduce` is the entry point: it regenerates every table, CSV and
# figure under results/, byte-identically. A clean `git diff` after running it
# means no number moved.

# Prefer the project venv when it exists, so `make` works without activating.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: help reproduce test figures install clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

reproduce: ## Run the full offline pipeline; regenerate every artifact
	$(PYTHON) -m lure.cli reproduce

test: ## Run the test suite (offline, FakeLLM only, never the network)
	$(PYTHON) -m pytest -q

figures: ## Regenerate only the figures
	$(PYTHON) -c "from lure.eval.report import run_benchmark; \
from lure.redteam.ood_eval import run_ood_comparison; \
from lure.viz.figures import generate_all_figures; \
print('\n'.join(str(p) for p in generate_all_figures(run_benchmark(), run_ood_comparison())))"

install: ## Create a venv and install the package with dev dependencies
	python3 -m venv .venv
	./.venv/bin/pip install --quiet --upgrade pip
	./.venv/bin/pip install --quiet -r requirements-dev.txt
	./.venv/bin/pip install --quiet -e .
	@echo "Installed. Run: make reproduce"

clean: ## Remove generated artifacts and caches
	rm -rf results/figures results/RESULTS.md results/metrics.csv
	rm -rf .pytest_cache **/__pycache__ *.egg-info
