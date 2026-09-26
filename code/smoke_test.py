"""
Standalone smoke test. Verifies core components without running the full suite.

    python smoke_test.py
"""
import sys
sys.path.insert(0, '.')
from pvi_experiments import smoke_test

if __name__ == '__main__':
    smoke_test()
    print('OK')
