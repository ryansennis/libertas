#!/usr/bin/env python3
"""
Test runner script for libertas project.

Usage:
    python scripts/test.py [options]

Examples:
    python scripts/test.py                     # Run all tests
    python scripts/test.py --quick             # Run unit tests only
    python scripts/test.py --unit              # Run unit tests
    python scripts/test.py --integration       # Run integration tests
    python scripts/test.py --e2e               # Run E2E tests
    python scripts/test.py --coverage          # Run with coverage report
    python scripts/test.py --filter worker     # Run tests matching "worker"
    python scripts/test.py --unit --coverage   # Unit tests with coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(
    test_type="all",
    coverage=False,
    filter_pattern=None,
    verbose=False,
    failfast=False,
    show_output=False
):
    """
    Run pytest with specified configuration.

    Args:
        test_type: Type of tests to run (all, quick, unit, integration, e2e)
        coverage: Whether to generate coverage report
        filter_pattern: Pattern to filter test names
        verbose: Verbose output
        failfast: Stop on first failure
        show_output: Show test output (don't capture)
    """
    # Base pytest command
    cmd = ["pytest", "src/libertas/tests/"]

    # Add test marker based on type
    if test_type == "quick" or test_type == "unit":
        cmd.extend(["-m", "unit"])
        print("🧪 Running unit tests...")
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
        print("🔗 Running integration tests...")
    elif test_type == "e2e":
        cmd.extend(["-m", "e2e"])
        print("🌍 Running E2E tests...")
    else:  # all
        cmd.extend(["-m", "unit or integration or e2e"])
        print("🚀 Running all tests...")

    # Add filter pattern if specified
    if filter_pattern:
        cmd.extend(["-k", filter_pattern])
        print(f"🔍 Filtering by: {filter_pattern}")

    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=libertas",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
        print("📊 Coverage reporting enabled")
    else:
        cmd.append("--no-cov")

    # Add verbosity
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")

    # Add failfast
    if failfast:
        cmd.append("-x")

    # Add output control
    if show_output:
        cmd.append("-s")

    # Add traceback style
    cmd.append("--tb=short")

    print(f"📝 Command: {' '.join(cmd)}\n")

    # Run pytest
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            print("\n✅ All tests passed!")
            if coverage:
                print("📊 Coverage report: htmlcov/index.html")
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")

        return result.returncode

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


def list_tests(test_type="all", filter_pattern=None):
    """List available tests without running them."""
    cmd = ["pytest", "src/libertas/tests/", "--collect-only", "-q"]

    if test_type == "quick" or test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "e2e":
        cmd.extend(["-m", "e2e"])
    else:
        cmd.extend(["-m", "unit or integration or e2e"])

    if filter_pattern:
        cmd.extend(["-k", filter_pattern])

    print(f"📋 Listing {test_type} tests...\n")
    subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for libertas project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         Run all tests
  %(prog)s --quick                 Run unit tests only (fast)
  %(prog)s --unit                  Run unit tests
  %(prog)s --integration           Run integration tests
  %(prog)s --e2e                   Run end-to-end tests
  %(prog)s --coverage              Run all tests with coverage
  %(prog)s --filter worker         Run tests matching "worker"
  %(prog)s --unit --coverage       Unit tests with coverage
  %(prog)s --list                  List available tests
  %(prog)s -x                      Stop on first failure
  %(prog)s -s                      Show test output (don't capture)
        """
    )

    # Test type selection (mutually exclusive)
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--all",
        action="store_const",
        const="all",
        dest="test_type",
        help="Run all tests (default)"
    )
    type_group.add_argument(
        "--quick",
        action="store_const",
        const="quick",
        dest="test_type",
        help="Run unit tests only (fastest)"
    )
    type_group.add_argument(
        "--unit",
        action="store_const",
        const="unit",
        dest="test_type",
        help="Run unit tests"
    )
    type_group.add_argument(
        "--integration",
        action="store_const",
        const="integration",
        dest="test_type",
        help="Run integration tests"
    )
    type_group.add_argument(
        "--e2e",
        action="store_const",
        const="e2e",
        dest="test_type",
        help="Run end-to-end tests"
    )

    # Other options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--filter",
        "-k",
        metavar="PATTERN",
        help="Filter tests by name pattern (e.g., 'worker' or 'test_market')"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests without running them"
    )
    parser.add_argument(
        "-x",
        "--failfast",
        action="store_true",
        help="Stop on first test failure"
    )
    parser.add_argument(
        "-s",
        "--show-output",
        action="store_true",
        help="Show test output (don't capture stdout/stderr)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    # Set default
    parser.set_defaults(test_type="all")

    args = parser.parse_args()

    # List tests if requested
    if args.list:
        list_tests(args.test_type, args.filter)
        return 0

    # Run tests
    return run_tests(
        test_type=args.test_type,
        coverage=args.coverage,
        filter_pattern=args.filter,
        verbose=args.verbose,
        failfast=args.failfast,
        show_output=args.show_output
    )


if __name__ == "__main__":
    sys.exit(main())
