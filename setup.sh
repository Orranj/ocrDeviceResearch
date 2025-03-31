#!/bin/bash

# Update and install system dependencies
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip tesseract-ocr espeak

# Create a virtual environment
python3 -m venv env
source env/bin/activate

echo "Installing system dependencies..."
xargs -a apt_requirements.txt sudo apt install -y

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete! Run 'source env/bin/activate' and 'python main.py' to start."
