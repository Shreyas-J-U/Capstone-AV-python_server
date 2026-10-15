import sys
import unittest
from pathlib import Path


def run_all_checks():
    print("=" * 80)
    print("AUTONOMOUS VEHICLE SYSTEM - COMPREHENSIVE HEALTH CHECK & VERIFICATION")
    print("=" * 80)
    print("Evaluating all modules: Protocol, Environment, Perception, Tracking,")
    print("Trajectory Prediction, and Safe Distance Estimation.")
    print("=" * 80)

    # 1. Discover and run all unit tests
    test_dir = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(test_dir),
        pattern="test_*.py",
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 80)
    print("SYSTEM HEALTH EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run   : {result.testsRun}")
    print(f"Failures          : {len(result.failures)}")
    print(f"Errors            : {len(result.errors)}")

    if result.wasSuccessful():
        print("RESULT: ALL TESTS PASSED! The codebase is healthy with NO errors.")
        print("=" * 80)
        return 0
    else:
        print("RESULT: ERRORS FOUND! Please inspect the traceback above.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())
