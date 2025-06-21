#!/bin/bash

# Setup script for IP updater service
set -e

echo "Setting up IP updater service..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Don't run this script as root. It will use sudo when needed."
    exit 1
fi

# Get the current user
USER=$(whoami)
REPO_NAME="Username2481632/KaMS-Club-Bot"

echo "Current user: $USER"

# Prompt for GitHub repository
read -p "Enter your GitHub repository (format: username/repo-name) [$REPO_NAME]: " input_repo
if [ ! -z "$input_repo" ]; then
    REPO_NAME="$input_repo"
fi

# Prompt for GitHub token
echo ""
echo "You need to create a GitHub Fine-grained Personal Access Token."
echo "Go to: https://github.com/settings/personal-access-tokens/new"
echo ""
echo "Required settings:"
echo "  - Resource owner: Select your username or organization"
echo "  - Repository access: Selected repositories -> Choose this repository"
echo "  - Repository permissions:"
echo "    * Actions: Write"
echo "    * Metadata: Read" 
echo "    * Secrets: Write"
echo ""
read -s -p "Enter your GitHub Fine-grained Personal Access Token: " GITHUB_TOKEN
echo ""

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GitHub token is required"
    exit 1
fi

# Install PyNaCl dependency for secret encryption
echo "Installing PyNaCl dependency..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Save GitHub token to file
echo "$GITHUB_TOKEN" > ~/.github_token
chmod 600 ~/.github_token
echo "GitHub token saved to ~/.github_token"

# Update service file with correct repository
sed -i "s|YourUsername/KaMS-Club-Bot|$REPO_NAME|g" ip-updater@.service

# Copy service files to systemd directory
echo "Installing systemd service files..."
sudo cp ip-updater@.service /etc/systemd/system/
sudo cp ip-updater@.timer /etc/systemd/system/

# Reload systemd and enable timer
echo "Enabling and starting systemd timer..."
sudo systemctl daemon-reload
sudo systemctl enable ip-updater@$USER.timer
sudo systemctl start ip-updater@$USER.timer

# Show status
echo ""
echo "Setup complete! Service status:"
sudo systemctl status ip-updater@$USER.timer --no-pager

echo ""
echo "To check logs: journalctl -u ip-updater@$USER.service -f"
echo "To stop timer: sudo systemctl stop ip-updater@$USER.timer"
echo "To disable timer: sudo systemctl disable ip-updater@$USER.timer"

echo ""
echo "The service will check for IP changes every 10 minutes."
echo "You can test it manually by running: python3 ip_updater.py"
