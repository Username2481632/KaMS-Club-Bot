#!/usr/bin/env python3
"""
Logs to the bot's log each time the Pi's supply voltage dropped too low, as the kernel reports it.

kams-bot-undervoltage@.timer runs it every minute. The journal cursor it keeps in its runtime
directory starts each run where the last one stopped and is gone after a reboot, so each boot is
read from its own beginning. The notifier passes the records on to the bot's owner.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

os.environ["TZ"] = "UTC"
time.tzset()

CURSOR_FILE = Path(os.environ["RUNTIME_DIRECTORY"]) / "cursor"
LOG_FILE = Path.home() / "KaMS-Club-Bot/bot.log"
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"  # The bot's own


def main() -> None:
    journal = subprocess.run(
        [
            "journalctl",
            "--dmesg",
            "--boot",
            f"--cursor-file={CURSOR_FILE}",
            "--grep=Undervoltage detected",
            "--output=cat",
            "--no-pager",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # journalctl exits 1, saying nothing, when no new entry matches.
    if journal.returncode == 1 and not journal.stderr:
        return
    if journal.returncode != 0:
        raise RuntimeError(f"journalctl failed: {journal.stderr.strip()}")
    warnings = journal.stdout.splitlines()
    logger = logging.getLogger("miphael")
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(handler)
    for _ in warnings:
        logger.warning(
            "The power supply's voltage dropped too low (the kernel's under-voltage warning)."
        )


if __name__ == "__main__":
    main()
