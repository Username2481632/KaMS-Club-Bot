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

echo "Current user: $USER"

# Source shared constants
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$SCRIPT_DIR/constants.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/constants.env"
else
    echo "constants.env not found; expected at $SCRIPT_DIR/constants.env"
    exit 1
fi

# Create config directory for the bot
mkdir -p "$BOT_CONFIG_DIR"
chmod 700 "$BOT_CONFIG_DIR"

# Load saved repository name if it exists
if [ -f "$REPO_CONFIG_FILE" ]; then
    SAVED_REPO=$(cat "$REPO_CONFIG_FILE" 2>/dev/null || echo "")
    if [ ! -z "$SAVED_REPO" ]; then
        echo "Previously used repository: $SAVED_REPO"
        read -p "Use this repository? (y/n) [y]: " use_saved
        if [ -z "$use_saved" ] || [ "$use_saved" = "y" ] || [ "$use_saved" = "Y" ]; then
            REPO_NAME="$SAVED_REPO"
        fi
    fi
fi

# Prompt for GitHub repository if not set
if [ -z "$REPO_NAME" ]; then
    while [ -z "$REPO_NAME" ]; do
        read -p "Enter your GitHub repository (format: username/repo-name): " REPO_NAME
        if [ -z "$REPO_NAME" ]; then
            echo "Error: Repository name is required"
        elif [[ ! "$REPO_NAME" =~ ^[^/]+/[^/]+$ ]]; then
            echo "Error: Repository name must be in format 'username/repo-name'"
            REPO_NAME=""
        fi
    done
fi

# Save repository name for future use
echo "$REPO_NAME" > "$REPO_CONFIG_FILE"
chmod 600 "$REPO_CONFIG_FILE"
echo "Repository name saved: $REPO_NAME"

# Load saved GitHub token if it exists
if [ -f "$GITHUB_TOKEN_FILE" ]; then
    SAVED_TOKEN=$(cat "$GITHUB_TOKEN_FILE" 2>/dev/null || echo "")
    if [ ! -z "$SAVED_TOKEN" ] && [[ "$SAVED_TOKEN" == github_pat_* ]]; then
        echo "Previously used GitHub token found."
        read -p "Use the existing GitHub token? (y/n) [y]: " use_saved_token
        if [ -z "$use_saved_token" ] || [ "${use_saved_token,,}" = "y" ]; then
            GITHUB_TOKEN="$SAVED_TOKEN"
        fi
    elif [ ! -z "$SAVED_TOKEN" ]; then
        echo "Warning: Saved token does not appear to be a valid GitHub fine-grained token (should start with 'github_pat_')."
    fi
fi

# Prompt for GitHub token if not set
if [ -z "$GITHUB_TOKEN" ]; then
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
fi

# Save GitHub token to file
echo "$GITHUB_TOKEN" > "$GITHUB_TOKEN_FILE"
chmod 600 "$GITHUB_TOKEN_FILE"
echo "GitHub token saved to $GITHUB_TOKEN_FILE"

# Update service file with correct repository
sed -i "s|PLACEHOLDER_REPO_NAME|$REPO_NAME|g" ip-updater@.service

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
