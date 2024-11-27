import unittest
import sys
import os
import pytest
import coverage

def setup_path():
    """Add the parent directory to the Python path"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

def run_tests():
    """Run the test suite with coverage"""
    # Setup path
    setup_path()
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    # Run pytest suite
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytest_args = [
        test_dir,
        '-v',  # verbose
        '--tb=short',  # shorter traceback format
        '--strict-markers',
        '-ra',  # show extra test summary
    ]
    
    if pytest.main(pytest_args) == 0:
        # Generate coverage report separately
        cov.report(show_missing=True)
        cov.html_report(directory='coverage_html')
        return True
    return False
def print_summary(success):
    """Print test execution summary"""
    print("\n" + "="*50)
    if success:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed!")
    print("="*50 + "\n")
    
    print("Test Coverage Report generated in: coverage_html/index.html")
    print("\nRun 'python -m http.server' in the coverage_html directory to view it")

if __name__ == '__main__':
    try:
        success = run_tests()
        print_summary(success)
        sys.exit(not success)
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during test execution: {e}")
        sys.exit(1)