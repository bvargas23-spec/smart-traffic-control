"""
Test API Debugger

This script helps debug failing tests in test_api.py by running each test function
individually and reporting which ones pass and which ones fail.
"""

import os
import sys
import traceback
from dotenv import load_dotenv # type: ignore

# Add the project root to the path so we can import modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Load the API key from .env
load_dotenv()
API_KEY = os.getenv("TOMTOM_API_KEY")

# Import the test functions from test_api.py
try:
    from test_api import (
        test_traffic_flow,
        test_traffic_incidents,
        test_intersection_approaches,
        test_traffic_summary
    )
    from src.api.tomtom_client import TomTomClient
except ImportError as e:
    print(f"Error importing test functions: {e}")
    sys.exit(1)

def run_test(test_func, client):
    """Run a test function and report if it passes or fails."""
    print(f"\n{'=' * 60}")
    print(f"Running test: {test_func.__name__}")
    print(f"{'=' * 60}")
    
    try:
        result = test_func(client)
        if result:
            print(f"✅ {test_func.__name__} PASSED")
            return True
        else:
            print(f"❌ {test_func.__name__} FAILED (returned False)")
            return False
    except Exception as e:
        print(f"❌ {test_func.__name__} FAILED with exception:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests individually."""
    if not API_KEY:
        print("❌ No API key found. Please set TOMTOM_API_KEY in your .env file.")
        return
    
    # Create TomTom client
    client = TomTomClient(api_key=API_KEY)
    
    # Define the tests to run
    tests = [
        test_traffic_flow,
        test_traffic_incidents,
        test_intersection_approaches,
        test_traffic_summary
    ]
    
    # Run each test and track results
    results = []
    
    for test in tests:
        results.append(run_test(test, client))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for i, test in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{status}: {test.__name__}")
    
    passed = sum(results)
    print(f"\nPASSED: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")

if __name__ == "__main__":
    main()