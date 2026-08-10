"""Reproducible, headless QwenPaw framework-overhead measurements.

The package deliberately lives outside :mod:`qwenpaw`: it observes the
installed runtime as an external client and does not change production APIs.
"""

SCHEMA_VERSION = "1"

__all__ = ["SCHEMA_VERSION"]

