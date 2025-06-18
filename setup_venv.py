#!/usr/bin/env python3
"""
FinSight Virtual Environment Setup Script

This script sets up a virtual environment for the FinSight project with all dependencies.
It works on Windows, macOS, and Linux.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, shell=True):
    """Run a command and handle errors"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro} OK")
    return True


def setup_virtual_environment():
    """Set up the virtual environment"""
    print("Setting up FinSight virtual environment...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    venv_path = Path(".venv")
    if venv_path.exists():
        print("Virtual environment already exists. Removing...")
        if platform.system() == "Windows":
            run_command("rmdir /s /q .venv", shell=True)
        else:
            run_command("rm -rf .venv", shell=True)
    
    print("Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv .venv"):
        return False
    
    # Determine activation script path
    if platform.system() == "Windows":
        activate_script = ".venv\\Scripts\\activate"
        pip_path = ".venv\\Scripts\\pip"
        python_path = ".venv\\Scripts\\python"
    else:
        activate_script = ".venv/bin/activate"
        pip_path = ".venv/bin/pip"
        python_path = ".venv/bin/python"
    
    # Upgrade pip
    print("Upgrading pip...")
    if not run_command(f"{python_path} -m pip install --upgrade pip"):
        return False
    
    # Install requirements
    print("Installing requirements...")
    if not run_command(f"{pip_path} install -r requirements.txt"):
        return False
    
    # Install package in development mode
    print("Installing FinSight in development mode...")
    if not run_command(f"{pip_path} install -e ."):
        return False
    
    # Create necessary directories
    print("Creating output directories...")
    directories = ["data", "outputs", "outputs/reports", "outputs/charts"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("\n[SUCCESS] FinSight virtual environment setup complete!")
    print("\nTo activate the environment:")
    if platform.system() == "Windows":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")
    
    print("\nTo run FinSight:")
    print("  python finsight_system.py")
    print("  or")
    print("  finsight  # if installed with entry point")
    
    print("\nOptional: Set up API keys")
    print("- The system works with defaults for basic functionality")
    print("- For enhanced features, set environment variables:")
    print("  * OPENAI_API_KEY for GPT models")
    print("  * ANTHROPIC_API_KEY for Claude verification")
    print("  * FINNHUB_API_KEY for enhanced market data")
    print("  * EDGAR_USER_AGENT if you need custom SEC compliance info")
    print("- You can create a .env file or set them in your system")
    
    return True


if __name__ == "__main__":
    success = setup_virtual_environment()
    sys.exit(0 if success else 1) 