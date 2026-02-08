#!/bin/bash

# Security Analysis Dashboard Setup Script

echo "================================================"
echo "Security Performance & Risk Analysis Dashboard"
echo "Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python 3.8 or higher
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python version is compatible"
else
    echo "✗ Python 3.8 or higher is required"
    exit 1
fi

echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "✗ Failed to activate virtual environment"
    exit 1
fi

echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Pip upgraded"
else
    echo "✗ Failed to upgrade pip"
fi

echo ""

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

echo ""
echo "================================================"
echo "Setup completed successfully!"
echo "================================================"
echo ""
echo "To run the dashboard:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the dashboard:"
echo "   streamlit run security_analysis_dashboard.py"
echo ""
echo "3. Open your browser and navigate to:"
echo "   http://localhost:8501"
echo ""
echo "================================================"
