"""
Test runner for DFormer Jittor implementation
"""

import os
import sys
import unittest
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Running DFormer Jittor Test Suite")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    return success


def run_specific_test(test_module):
    """Run a specific test module."""
    print(f"Running tests from {test_module}")
    print("=" * 40)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


def run_quick_tests():
    """Run a subset of quick tests."""
    print("Running quick tests...")
    print("=" * 40)
    
    # Import test modules
    from test_models import TestDFormerModels
    from test_data import TestDataLoading
    from test_losses import TestLossFunctions
    
    # Create test suite with selected tests
    suite = unittest.TestSuite()
    
    # Add quick model tests
    suite.addTest(TestDFormerModels('test_model_creation'))
    suite.addTest(TestDFormerModels('test_model_forward'))
    
    # Add quick data tests
    suite.addTest(TestDataLoading('test_transforms'))
    
    # Add quick loss tests
    suite.addTest(TestLossFunctions('test_cross_entropy_loss'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='DFormer Jittor Test Runner')
    parser.add_argument('--test', help='Run specific test module')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            success = run_specific_test(args.test)
        elif args.quick:
            success = run_quick_tests()
        else:
            success = run_all_tests()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
