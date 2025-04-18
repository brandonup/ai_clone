#!/bin/bash

# Firestore Data Puller Shell Script
# This script provides a simple interface to pull clone data from Firestore

echo "========================================"
echo "Firestore Data Puller"
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
chmod +x pull_firestore_data.py

# Display menu
echo "Select an option:"
echo "1. Pull all clone data from Firestore"
echo "2. Pull a specific clone by ID"
echo "3. Pull all clone data and update diagnose_rag.py"
echo "4. Exit"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Pulling all clone data from Firestore..."
        python3 pull_firestore_data.py
        ;;
    2)
        echo "Enter the clone ID:"
        read -p "> " clone_id
        echo "Pulling clone $clone_id from Firestore..."
        python3 pull_firestore_data.py --clone-id "$clone_id"
        ;;
    3)
        echo "Pulling all clone data and updating diagnose_rag.py..."
        python3 pull_firestore_data.py --update-script
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

# Check if the log file exists and display it
if [ -f "firestore_pull.log" ]; then
    echo ""
    echo "Pull log saved to firestore_pull.log"
    echo ""
    echo "To view the full log, run:"
    echo "cat firestore_pull.log"
fi

echo "========================================"
echo "Firestore Data Puller Complete"
echo "========================================"
