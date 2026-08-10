"""Allow ``python -m benchmarks.overhead``."""

from .qwenpaw_overhead.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

