#!/bin/bash

# RAG Diagnostics Shell Script
# This script runs the Python diagnostic tool and displays the results

echo "========================================"
echo "Starting RAG Diagnostics"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if the virtual environment exists
if [ -d "ai_clone_env" ]; then
    echo "Activating virtual environment..."
    source ai_clone_env/bin/activate
fi

# Run the diagnostic script
echo "Running diagnostic tests..."
python3 diagnose_rag.py

# Check if the log file exists and display it
if [ -f "rag_diagnostics.log" ]; then
    echo ""
    echo "Diagnostic log saved to rag_diagnostics.log"
    echo ""
    echo "To view the full log, run:"
    echo "cat rag_diagnostics.log"
fi

echo "========================================"
echo "RAG Diagnostics Complete"
echo "========================================"
