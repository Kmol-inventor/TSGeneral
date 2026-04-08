"""
Entry point for `python -m tsgeneral` and the `tsgeneral` CLI command.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="tsgeneral",
        description="TSGeneral - Time-series inspector with filter pipelines",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Launch a demo inspector with random data",
    )

    args = parser.parse_args()

    if args.version:
        from tsgeneral import __version__
        print(f"tsgeneral {__version__}")
        return

    if args.demo:
        _run_demo()
        return

    # Default: print usage
    parser.print_help()


def _run_demo():
    """Launch the inspector with synthetic demo data."""
    import numpy as np
    from tsgeneral import Inspector, Pipeline

    np.random.seed(42)
    n_trials, n_samples = 5, 256
    fs = 128.0
    t = np.arange(n_samples) / fs

    # Synthetic EEG-like data: sine + noise
    data = np.array([
        np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(n_samples)
        for _ in range(n_trials)
    ])

    pipeline = Pipeline()
    pipeline.add_stage("Raw")
    pipeline.add_stage("Smoothed", lambda x: np.convolve(x, np.ones(5) / 5, mode="same"))

    inspector = Inspector(data, pipeline, sample_rate=fs)
    inspector.run()


if __name__ == "__main__":
    main()
