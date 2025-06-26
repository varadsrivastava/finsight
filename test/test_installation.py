#!/usr/bin/env python3
"""
FinSight Installation Test Script

This script tests if the FinSight installation in the virtual environment is working correctly.
"""

import sys
import os
import importlib
from pathlib import Path


def test_python_version():
    """Test Python version compatibility"""
    print("Testing Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"[FAIL] Python version {version.major}.{version.minor}.{version.micro} is too old")
        print("   Required: Python 3.8 or higher")
        return False
    print(f"[OK] Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def test_virtual_environment():
    """Test if we're running in a virtual environment"""
    print("\nTesting virtual environment...")
    
    # Check if we're in a virtual environment
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    
    if in_venv:
        print("[OK] Running in virtual environment")
        print(f"   Virtual env path: {sys.prefix}")
        return True
    else:
        print("[WARN] Not running in virtual environment")
        print("   Consider activating the virtual environment:")
        print("   Windows: .venv\\Scripts\\activate")
        print("   Unix/Linux/macOS: source .venv/bin/activate")
        return False


def test_core_dependencies():
    """Test if core dependencies are installed"""
    print("\nTesting core dependencies...")
    
    dependencies = [
        "autogen",
        "openai", 
        "anthropic",
        "yfinance",
        "pandas",
        "numpy",
        "matplotlib",
        "requests",
        "chromadb",
        "pydantic",
        "dotenv"
    ]
    
    failed = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"[OK] {dep}")
        except ImportError:
            print(f"[FAIL] {dep} - not installed")
            failed.append(dep)
    
    if failed:
        print(f"\n[FAIL] Missing dependencies: {', '.join(failed)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n[OK] All core dependencies are installed")
    return True


def test_finsight_package():
    """Test if FinSight package is properly installed"""
    print("\nTesting FinSight package...")
    
    try:
        import finsight
        print("[OK] FinSight package imported successfully")
        print(f"   Version: {getattr(finsight, '__version__', 'unknown')}")
        
        # Test main components
        from finsight import FinSightOrchestrator, FinSightConfig
        print("[OK] Main components imported successfully")
        
        return True
    except ImportError as e:
        print(f"[FAIL] FinSight package import failed: {e}")
        print("   Run: pip install -e .")
        return False


def test_configuration():
    """Test configuration setup"""
    print("\nTesting configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("[OK] .env file exists")
    else:
        print("[WARN] .env file not found")
        print("   Optional: Set up API keys for enhanced features")
        print("   - Create a .env file or set environment variables")
        print("   - OPENAI_API_KEY, ANTHROPIC_API_KEY, FINNHUB_API_KEY, etc.")
    
    # Check if required directories exist
    required_dirs = ["data", "outputs", "outputs/reports", "outputs/charts"]
    missing_dirs = []
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"[OK] Directory {directory} exists")
        else:
            print(f"[WARN] Directory {directory} missing")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"   Creating missing directories...")
        for directory in missing_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory}")
    
    return True


def test_basic_functionality():
    """Test basic FinSight functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from finsight import FinSightConfig
        config = FinSightConfig()
        print("[OK] FinSight configuration loaded")
        
        from finsight import FinSightOrchestrator
        # Don't initialize fully to avoid API key requirements
        print("[OK] FinSight orchestrator class available")
        
        return True
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        return False


def main():
    """Run all installation tests"""
    print("FinSight Installation Test\n")
    print("=" * 50)
    
    tests = [
        test_python_version,
        test_virtual_environment, 
        test_core_dependencies,
        test_finsight_package,
        test_configuration,
        test_basic_functionality
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! FinSight is ready to use.")
        print("\nNext steps:")
        print("1. Run: python finsight_system.py")
        print("2. Optional: Set API keys for enhanced features (create .env file or set environment variables)")
        return True
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Please address the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 