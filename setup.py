#!/usr/bin/env python3
"""Quick setup script for VLN Navigation project."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("VLN Navigation Project Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("Failed to install requirements. Please check your Python environment.")
        sys.exit(1)
    
    # Create necessary directories
    print("\nCreating project directories...")
    directories = [
        "data", "checkpoints", "outputs", "logs", "assets"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run basic tests
    print("\nRunning basic tests...")
    if not run_command("python -c \"from src.models import CLIPVLNModel; print('Model import successful')\"", "Testing model imports"):
        print("Warning: Model imports failed. Check your installation.")
    
    # Test demo
    print("\nTesting demo...")
    if not run_command("python -c \"from demo.app import create_streamlit_app; print('Demo import successful')\"", "Testing demo imports"):
        print("Warning: Demo imports failed. Check your installation.")
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Run the simple demo: python 0573.py")
    print("2. Start training: python scripts/train.py")
    print("3. Launch interactive demo: streamlit run demo/app.py")
    print("4. Run tests: pytest tests/")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
