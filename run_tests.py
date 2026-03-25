"""
Ajentify Testing Framework — CLI Entry Point

Usage:
    python run_tests.py                  # run all tests (parallel)
    python run_tests.py --test my_test   # run a single test by name
"""

from ajentify_testing.runner import main

if __name__ == "__main__":
    main()
