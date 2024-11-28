import unittest
import sys
import os

def run_tests():
    # Add the parent directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(current_dir, 'gen_tests')
    
    # Use pattern that matches test files in subdirectories
    suite = loader.discover(start_dir, pattern='test_*.py', top_level_dir=current_dir)
    
    # Check if any tests were found
    if suite.countTestCases() == 0:
        print("No tests were discovered!")
        return False

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_tests()  # Actually run the tests when the script is executed