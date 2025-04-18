#!/bin/bash

# Undo RAG Changes Script
# This script reverts changes made by the fix_rag_issues.py script

echo "========================================"
echo "Undo RAG Changes"
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

# Function to restore .env file from backup
restore_env_file() {
    echo "Looking for .env backup files..."
    
    # Find the most recent .env backup file
    BACKUP_DIR="backend"
    LATEST_BACKUP=$(ls -t ${BACKUP_DIR}/.env.backup.* 2>/dev/null | head -1)
    
    if [ -z "$LATEST_BACKUP" ]; then
        echo "No .env backup files found in ${BACKUP_DIR}/"
        return 1
    fi
    
    echo "Found backup file: $LATEST_BACKUP"
    echo "Restoring from backup..."
    
    # Restore the backup
    cp "$LATEST_BACKUP" "${BACKUP_DIR}/.env"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully restored .env file from backup"
        return 0
    else
        echo "❌ Failed to restore .env file from backup"
        return 1
    fi
}

# Function to list all available backups
list_backups() {
    echo "Available .env backup files:"
    ls -lt backend/.env.backup.* 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "No backup files found."
        return 1
    fi
    
    return 0
}

# Function to restore a specific backup
restore_specific_backup() {
    echo "Enter the full path of the backup file to restore:"
    read -p "> " backup_file
    
    if [ ! -f "$backup_file" ]; then
        echo "❌ File not found: $backup_file"
        return 1
    fi
    
    echo "Restoring from $backup_file..."
    cp "$backup_file" "backend/.env"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully restored .env file from $backup_file"
        return 0
    else
        echo "❌ Failed to restore .env file from $backup_file"
        return 1
    fi
}

# Function to clean up log files
clean_logs() {
    echo "Cleaning up log files..."
    
    # Remove diagnostic log files
    if [ -f "rag_diagnostics.log" ]; then
        rm "rag_diagnostics.log"
        echo "Removed rag_diagnostics.log"
    fi
    
    # Remove fix log files
    if [ -f "rag_fixes.log" ]; then
        rm "rag_fixes.log"
        echo "Removed rag_fixes.log"
    fi
    
    echo "✅ Log files cleaned up"
    return 0
}

# Display menu
echo "Select an option:"
echo "1. Restore .env file from the most recent backup"
echo "2. List all available backups"
echo "3. Restore from a specific backup"
echo "4. Clean up log files"
echo "5. Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        restore_env_file
        ;;
    2)
        list_backups
        
        # Ask if user wants to restore a specific backup
        echo ""
        read -p "Do you want to restore a specific backup? (y/n): " restore_specific
        if [ "$restore_specific" = "y" ]; then
            restore_specific_backup
        fi
        ;;
    3)
        restore_specific_backup
        ;;
    4)
        clean_logs
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo "========================================"
echo "Undo RAG Changes Complete"
echo "========================================"
