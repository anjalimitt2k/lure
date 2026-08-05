"""Figure generation for the Lure write-up.

Deterministic and offline, like everything else in the pipeline: fixed style,
fixed figure size and DPI, no randomness, and no timestamps written into the
PNGs, so repeated runs produce byte-identical files.
"""

from lure.viz.figures import DEFAULT_FIGURE_DIR, generate_all_figures

__all__ = ["DEFAULT_FIGURE_DIR", "generate_all_figures"]
