#!/usr/bin/env python3
"""
IP Updater for Raspberry Pi
Checks for public IP changes and updates GitHub Actions secrets accordingly.
"""
import os
import sys
import json
import logging
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import Optional

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_TOKEN_FILE = os.getenv('GITHUB_TOKEN_FILE')
GITHUB_REPO = os.getenv('GITHUB_REPO')  # Format: "owner/repo"

# Read token from file if not provided in environment
if not GITHUB_TOKEN and GITHUB_TOKEN_FILE:
    try:
        with open(os.path.expanduser(GITHUB_TOKEN_FILE), 'r') as f:
            GITHUB_TOKEN = f.read().strip()
    except Exception as e:
        print(f"Error reading GitHub token from file: {e}")

IP_CACHE_FILE = Path.home() / '.ip_cache.json'
LOG_FILE = Path.home() / 'ip_updater.log'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ip_updater')


async def get_public_ip() -> Optional[str]:
    """Get the current public IP address."""
    ip_services = [
        'https://api.ipify.org',
        'https://api4.ipify.org',  # IPv4 only
        'https://icanhazip.com',
        'https://ifconfig.me/ip',
        'https://checkip.amazonaws.com'
    ]
    
    async with aiohttp.ClientSession() as session:
        for service in ip_services:
            try:
                async with session.get(service, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        ip = content.strip()
                        
                        # Validate that we got an IP address, not HTML
                        if ip and not ip.startswith('<!DOCTYPE') and not ip.startswith('<html'):
                            # Simple IP validation (basic check for IPv4/IPv6 format)
                            ipv4_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
                            ipv6_pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
                            
                            if re.match(ipv4_pattern, ip) or re.match(ipv6_pattern, ip) or ':' in ip:
                                logger.info(f"Got IP {ip} from {service}")
                                return ip
                        
                        logger.warning(f"Got invalid response from {service}: {ip[:100]}...")
                        
            except Exception as e:
                logger.warning(f"Failed to get IP from {service}: {e}")
                continue
    
    logger.error("Failed to get public IP from all services")
    return None


def load_cached_ip() -> Optional[str]:
    """Load the previously cached IP address."""
    try:
        if IP_CACHE_FILE.exists():
            with open(IP_CACHE_FILE, 'r') as f:
                data = json.load(f)
                return data.get('ip')
    except Exception as e:
        logger.warning(f"Failed to load cached IP: {e}")
    return None


def save_cached_ip(ip: str) -> None:
    """Save the current IP address to cache."""
    try:
        with open(IP_CACHE_FILE, 'w') as f:
            json.dump({'ip': ip}, f)
        logger.info(f"Cached IP: {ip}")
    except Exception as e:
        logger.error(f"Failed to cache IP: {e}")


async def update_github_secret(secret_name: str, secret_value: str) -> bool:
    """Update a GitHub Actions secret."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.error("GITHUB_TOKEN and GITHUB_REPO environment variables must be set")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/{secret_name}"
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'Content-Type': 'application/json'
    }

    # First, get the repository's public key for encryption
    async with aiohttp.ClientSession() as session:
        try:
            # Get public key
            key_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/public-key"
            async with session.get(key_url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to get public key: {response.status}")
                    return False
                
                key_data = await response.json()
                public_key = key_data['key']
                key_id = key_data['key_id']

            # Encrypt the secret value
            from nacl import encoding, public as nacl_public
            
            public_key_bytes = encoding.Base64Encoder.decode(public_key)
            public_key_obj = nacl_public.PublicKey(public_key_bytes)
            box = nacl_public.SealedBox(public_key_obj)
            encrypted = box.encrypt(secret_value.encode('utf-8'))
            encrypted_value = encoding.Base64Encoder.encode(encrypted).decode('utf-8')

            # Update the secret
            payload = {
                'encrypted_value': encrypted_value,
                'key_id': key_id
            }

            async with session.put(url, headers=headers, json=payload) as response:
                if response.status in [201, 204]:
                    logger.info(f"Successfully updated GitHub secret '{secret_name}'")
                    return True
                else:
                    logger.error(f"Failed to update secret: {response.status} - {await response.text()}")
                    return False

        except Exception as e:
            logger.error(f"Error updating GitHub secret: {e}")
            return False


async def trigger_workflow() -> bool:
    """Trigger the GitHub Actions workflow."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.error("GITHUB_TOKEN and GITHUB_REPO environment variables must be set")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/main.yml/dispatches"
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'Content-Type': 'application/json'
    }
    payload = {'ref': 'main'}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 204:
                    logger.info("Successfully triggered GitHub Actions workflow")
                    return True
                else:
                    logger.error(f"Failed to trigger workflow: {response.status} - {await response.text()}")
                    return False
        except Exception as e:
            logger.error(f"Error triggering workflow: {e}")
            return False


async def main():
    """Main function to check IP and update GitHub if needed."""
    logger.info("Starting IP update check...")

    # Get current public IP
    current_ip = await get_public_ip()
    if not current_ip:
        logger.error("Could not determine public IP address")
        sys.exit(1)

    # Load cached IP
    cached_ip = load_cached_ip()
    
    if current_ip == cached_ip:
        logger.info(f"IP unchanged: {current_ip}")
        return

    logger.info(f"IP changed from {cached_ip} to {current_ip}")

    # Update GitHub secret
    if await update_github_secret('HOST', current_ip):
        # Trigger workflow
        if await trigger_workflow():
            # Save new IP to cache
            save_cached_ip(current_ip)
            logger.info("Successfully updated IP and triggered deployment")
        else:
            logger.error("Failed to trigger workflow")
            sys.exit(1)
    else:
        logger.error("Failed to update GitHub secret")
        sys.exit(1)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
