#!/bin/bash
# FinSight Virtual Environment Setup Script for Unix/Linux/macOS
# This script creates and sets up a virtual environment for FinSight

echo "Setting up FinSight virtual environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $python_version"

# Run the Python setup script
python3 setup_venv.py
if [ $? -ne 0 ]; then
    echo "Error: Setup failed"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run FinSight:"
echo "  python finsight_system.py"
echo "" 