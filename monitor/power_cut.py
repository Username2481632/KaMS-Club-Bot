#!/usr/bin/env python3
"""
Logs to the bot's log how the Pi went down, when it came back without a clean shutdown.

kams-bot-power-record left a note at boot: the last time the Pi was up, and what the real-time
clock read. The notifier passes the record on to the bot's owner.
"""

import logging
import math
import os
import time
from pathlib import Path

os.environ["TZ"] = "UTC"
time.tzset()

NOTE_FILE = Path("/run/kams-bot/power-cut")
POWER_RESET_FILE = Path("/proc/device-tree/chosen/power/power_reset")
LOG_FILE = Path.home() / "KaMS-Club-Bot/bot.log"
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"  # The bot's own
# Bits of power_reset, the reason the Pi 5's power chip reset the board (Raspberry Pi
# documentation, "BCM2712 specific bootloader properties").
POWER_RESET_REASONS = [
    "over voltage",
    "under voltage",
    "over temperature",
    "the enable signal",
    "its watchdog",
]


def format_duration(seconds: float) -> str:
    minutes = math.ceil(seconds / 60)
    if minutes < 120:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes} min"
    days, hours = divmod(hours, 24)
    return f"{days} days {hours} h"


def main() -> None:
    # Either field is empty when it could not be read.
    last_up, _, rtc_at_boot = NOTE_FILE.read_text().rstrip("\n").partition(" ")
    with open("/proc/uptime") as uptime_file:
        came_back = time.time() - float(uptime_file.read().split()[0])

    if not last_up:
        message = "Came back without a clean shutdown; when it went down is unknown."
    else:
        # The last time the Pi was up is about a minute or less before it went down, so this
        # is the longest it can have been down.
        down_for = format_duration(came_back - int(last_up))
        if rtc_at_boot and int(rtc_at_boot) >= int(last_up):
            message = (
                f"Came back from going down without a clean shutdown, down for at most "
                f"{down_for}. It was not a power cut: the real-time clock kept its time."
            )
        else:
            message = f"Came back from a power cut, off for at most {down_for}."

    # Only a Pi 5 records why its power chip reset the board.
    power_reset = (
        int.from_bytes(POWER_RESET_FILE.read_bytes(), "big")
        if POWER_RESET_FILE.exists()
        else 0
    )
    reasons = [
        reason
        for bit, reason in enumerate(POWER_RESET_REASONS)
        if power_reset & (1 << bit)
    ]
    if reasons:
        message += f" The power chip reset the board for {' and '.join(reasons)}."

    logger = logging.getLogger("miphael")
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(handler)
    logger.warning(message)


if __name__ == "__main__":
    main()
