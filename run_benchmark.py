#!/usr/bin/env python3
"""
run_benchmark.py — Top-level entry point
=========================================
Usage:
  python3 run_benchmark.py                           # run all TCs, all descriptors, both phases
  python3 run_benchmark.py --tc TC-00                # single test case
  python3 run_benchmark.py --descriptor sift rift2   # specific descriptors
  python3 run_benchmark.py --phase 1                 # Phase 1 only (RANSAC baseline)
  python3 run_benchmark.py --phase 2                 # Phase 2 only (our MAGSAC++ approach)

Data must exist in the data/ directory. Results are written to results/.
"""
import os
import sys
import argparse

# Allow imports from pipeline/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(THIS_DIR, "pipeline")
sys.path.insert(0, PIPELINE_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Lunar Cross-Sensor Registration Benchmark"
    )
    parser.add_argument(
        "--tc", nargs="+", default=None,
        help="Test case IDs to run (e.g. TC-00 TC-01). Default: all."
    )
    parser.add_argument(
        "--descriptor", nargs="+", default=None,
        choices=["sift", "rift2"],
        help="Descriptors to evaluate. Default: both."
    )
    parser.add_argument(
        "--phase", nargs="+", type=int, default=None, choices=[1, 2],
        help="Phases to run (1=RANSAC baseline, 2=MAGSAC++). Default: both."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to test_cases.json. Default: auto-detected."
    )
    args = parser.parse_args()

    config_path = args.config or os.path.join(THIS_DIR, "test_cases.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] test_cases.json not found at {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("LUNAR CROSS-SENSOR IMAGE CORRESPONDENCE")
    print("Team: GradientZero")
    print("Phase 1: Makharia et al. 2025 pipeline (RANSAC baseline)")
    print("Phase 2: DEM Level-1 gate + MAGSAC++ (our approach)")
    print("=" * 70)
    print(f"  Test cases:  {args.tc or 'ALL'}")
    print(f"  Descriptors: {args.descriptor or 'ALL (sift, rift2)'}")
    print(f"  Phases:      {args.phase or 'BOTH (1 and 2)'}")
    print()

    from evaluation.benchmark import run_all
    run_all(
        tc_ids      = args.tc,
        descriptors = args.descriptor,
        phases      = args.phase,
        config_path = config_path,
    )


if __name__ == "__main__":
    main()
