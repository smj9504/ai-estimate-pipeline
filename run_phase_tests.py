#!/usr/bin/env python3
"""
Phase Test Runner - Simplified interface for running phase tests
"""
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tests.phases.cli import PhaseTestCLI


def main():
    """Main entry point for phase testing"""
    print("AI Estimate Pipeline - Phase Testing Framework")
    print("=" * 55)
    
    if len(sys.argv) == 1:
        # Interactive mode when no arguments provided
        print("\nAvailable commands:")
        print("  single    - Run a single phase test")
        print("  pipeline  - Run multiple phases in sequence") 
        print("  compare   - Compare different configurations")
        print("  scenario  - Run predefined test scenarios")
        print("  list-configs - Show available configurations")
        print("  list-results - Show recent test results")
        
        print("\nQuick Examples:")
        print("  python run_phase_tests.py single --phase 1")
        print("  python run_phase_tests.py pipeline --phases 0 1 2")
        print("  python run_phase_tests.py compare --phase 1 --compare-type models")
        print("  python run_phase_tests.py scenario validation_comparison")
        
        print("\nFor detailed help:")
        print("  python run_phase_tests.py --help")
        print("  python run_phase_tests.py single --help")
        print("  python run_phase_tests.py pipeline --help")
        
        return 0
    
    # Run CLI with provided arguments
    cli = PhaseTestCLI()
    return asyncio.run(cli.run())


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)