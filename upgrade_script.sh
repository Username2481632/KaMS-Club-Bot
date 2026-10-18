#!/bin/bash
set -e

# welcome_dm is a string now: an empty one sends no welcome.
python3 - <<'PYTHON'
import json
import os

with open("config.json") as file:
    config = json.load(file)
for guild_config in config.values():
    if guild_config["welcome_dm"] is None:
        guild_config["welcome_dm"] = ""
with open("config.json.tmp", "w") as file:
    json.dump(config, file, indent=2)
    file.flush()
    os.fsync(file.fileno())
os.replace("config.json.tmp", "config.json")
PYTHON

# The bot runs only with every file it keeps in place, fingerprinted: ban_requests.json never
# existed here, and --init accepts it and data.json as they are.
if [ ! -e ban_requests.json ]; then
  printf '{}' > ban_requests.json
fi
venv/bin/python3.12 src/bot.py --init

venv/bin/pip uninstall -y -q aiofiles
