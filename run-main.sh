#!/bin/bash

# Make this file executable: chmod +x run-main.sh
# Run it with: ./run-main.sh

clear

# === Define color codes ===
GREEN_BG="\033[42m"
WHITE_TEXT="\033[97m"
RED_BG="\033[41m"
RESET="\033[0m"

print_green() {
    echo ""
    echo -e "${GREEN_BG}${WHITE_TEXT}>>> $1${RESET}"
    echo ""
}

print_error() {
    echo ""
    echo -e "${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}"
    echo ""
    exit 12
}

# === Activate the virtual environment ===
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    print_error "Virtual environment not found. Please run setup-env.sh first."
fi

# === Run the main Python script ===
print_green "Running the main menu (main.py)..."
python -m src.main || print_error "main.py execution failed"

# === Done ===
print_green "Executi_
