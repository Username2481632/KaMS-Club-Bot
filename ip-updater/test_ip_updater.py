#!/usr/bin/env python3
"""
Test script for IP updater functionality
"""
import os
import asyncio
import sys
from pathlib import Path

# Add the project directory to the path
sys.path.append(str(Path(__file__).parent))

from ip_updater import get_public_ip, update_github_secret, trigger_workflow

async def test_ip_updater():
    """Test the IP updater functionality."""
    print("Testing IP updater functionality...")
    
    # Test getting public IP
    print("\n1. Testing public IP detection...")
    ip = await get_public_ip()
    if ip:
        print(f"✓ Public IP: {ip}")
    else:
        print("✗ Failed to get public IP")
        return False
    
    # Check environment variables
    print("\n2. Checking environment variables...")
    github_token = os.getenv('GITHUB_TOKEN')
    github_repo = os.getenv('GITHUB_REPO')
    
    if not github_token:
        # Try to read from file
        token_file = Path.home() / '.github_token'
        if token_file.exists():
            github_token = token_file.read_text().strip()
            os.environ['GITHUB_TOKEN'] = github_token
            print("✓ GitHub token loaded from file")
        else:
            print("✗ GITHUB_TOKEN not found in environment or ~/.github_token file")
            return False
    else:
        print("✓ GitHub token found in environment")
    
    if not github_repo:
        print("✗ GITHUB_REPO environment variable not set")
        print("  Set it with: export GITHUB_REPO='username/repo-name'")
        return False
    else:
        print(f"✓ GitHub repo: {github_repo}")
    
    # Test updating GitHub secret (but don't actually change it)
    print(f"\n3. Would update GitHub secret 'HOST' to: {ip}")
    print("   (Run the actual script to perform the update)")
    
    print("\n✓ All checks passed! The IP updater should work correctly.")
    return True

if __name__ == '__main__':
    try:
        success = asyncio.run(test_ip_updater())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
