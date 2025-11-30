#!/bin/bash

# Setup script for Pixeltable Demo
echo "🚀 Setting up Pixeltable Demo Environment..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install spacy language model
echo "📦 Downloading spacy language model..."
python -m spacy download en_core_web_sm

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the Streamlit app:"
echo "  streamlit run app.py"
echo ""
echo "To run Jupyter notebook:"
echo "  jupyter notebook pixeltable_demo.ipynb"
