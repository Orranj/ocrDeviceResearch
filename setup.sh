#!/bin/bash

# Update and install system dependencies
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip tesseract-ocr espeak

# Clone the repository
git clone https://github.com/Orranj/ocrDeviceResearch.git
cd ocrDeviceResearch

# Create a virtual environment
python3 -m venv env
source env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete! Run 'source env/bin/activate' and 'python main.py' to start."
