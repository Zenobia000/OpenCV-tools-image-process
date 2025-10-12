#!/bin/bash
# Quick Start Script for OpenCV Computer Vision Toolkit
# This script provides common Poetry commands for the project

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}OpenCV Computer Vision Toolkit${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}⚠️  Poetry is not installed!${NC}"
    echo "Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo -e "${GREEN}✅ Poetry detected: $(poetry --version)${NC}"
echo ""

# Function to show menu
show_menu() {
    echo "Select an option:"
    echo "1) Install dependencies (first time setup)"
    echo "2) Enter Poetry shell"
    echo "3) Run tests"
    echo "4) Run tests with coverage"
    echo "5) Start Jupyter Lab"
    echo "6) Format code (Black + isort)"
    echo "7) Lint code (Flake8)"
    echo "8) Show installed packages"
    echo "9) Update dependencies"
    echo "10) Clean environment"
    echo "0) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice: " choice

    case $choice in
        1)
            echo -e "${BLUE}📦 Installing dependencies...${NC}"
            poetry install --no-interaction
            echo -e "${GREEN}✅ Installation complete!${NC}"
            ;;
        2)
            echo -e "${BLUE}🐚 Entering Poetry shell...${NC}"
            echo "Type 'exit' to leave the shell"
            poetry shell
            ;;
        3)
            echo -e "${BLUE}🧪 Running tests...${NC}"
            poetry run pytest -v
            ;;
        4)
            echo -e "${BLUE}📊 Running tests with coverage...${NC}"
            poetry run pytest --cov=utils --cov-report=term --cov-report=html
            echo -e "${GREEN}✅ Coverage report generated in htmlcov/index.html${NC}"
            ;;
        5)
            echo -e "${BLUE}📓 Starting Jupyter Lab...${NC}"
            poetry run jupyter lab
            ;;
        6)
            echo -e "${BLUE}✨ Formatting code...${NC}"
            poetry run black utils/ tests/
            poetry run isort utils/ tests/
            echo -e "${GREEN}✅ Code formatted!${NC}"
            ;;
        7)
            echo -e "${BLUE}🔍 Linting code...${NC}"
            poetry run flake8 utils/ tests/
            ;;
        8)
            echo -e "${BLUE}📋 Installed packages:${NC}"
            poetry show
            ;;
        9)
            echo -e "${BLUE}🔄 Updating dependencies...${NC}"
            poetry update
            echo -e "${GREEN}✅ Dependencies updated!${NC}"
            ;;
        10)
            echo -e "${YELLOW}🧹 Cleaning environment...${NC}"
            read -p "Are you sure? This will remove .venv/ (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                rm -rf .venv
                rm -f poetry.lock
                echo -e "${GREEN}✅ Environment cleaned!${NC}"
                echo "Run option 1 to reinstall dependencies"
            fi
            ;;
        0)
            echo -e "${GREEN}👋 Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}⚠️  Invalid choice. Please try again.${NC}"
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
    clear
done
