#!/bin/bash

# RAG Issues Fixer Shell Script
# This script provides a simple interface to fix RAG issues

echo "========================================"
echo "RAG Issues Fixer"
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

# Make the Python script executable
chmod +x fix_rag_issues.py

# Display menu
echo "Select an option to fix:"
echo "1. API key issues"
echo "2. Ragie retrieval issues"
echo "3. Adaptive router issues"
echo "4. Rate limit issues"
echo "5. All issues"
echo "6. Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "Fixing API key issues..."
        python3 fix_rag_issues.py --api-keys
        ;;
    2)
        echo "Fixing Ragie retrieval issues..."
        python3 fix_rag_issues.py --ragie
        ;;
    3)
        echo "Fixing adaptive router issues..."
        python3 fix_rag_issues.py --router
        ;;
    4)
        echo "Fixing rate limit issues..."
        python3 fix_rag_issues.py --rate-limits
        ;;
    5)
        echo "Fixing all issues..."
        python3 fix_rag_issues.py --all
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

# Check if the log file exists and display it
if [ -f "rag_fixes.log" ]; then
    echo ""
    echo "Fix log saved to rag_fixes.log"
    echo ""
    echo "To view the full log, run:"
    echo "cat rag_fixes.log"
fi

echo "========================================"
echo "RAG Issues Fixer Complete"
echo "========================================"
