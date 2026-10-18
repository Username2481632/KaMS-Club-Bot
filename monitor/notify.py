#!/usr/bin/env python3
"""
Passes the warnings and errors in the bot's log on to the bot's owner, as a DM from the bot.

kams-bot-notify@.timer runs it every few minutes. Each run reads bot.log on from where the last
one stopped, so a burst of records arrives as one message, and a run that cannot reach Discord
leaves the records for the next. Errors are passed on from every logger, and warnings from the
bot's own and the Pi's: discord.py's warnings are about its own reconnects. A record that only
repeats one already reported that day, numbers aside, is left out, so a problem that persists is
reported once a day rather than every few minutes; the Pi's own records, a power cut or a drop in
its supply voltage, are each an event of their own and always reported.
"""

import datetime
import json
import os
import re
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

BOT_DIR = Path.home() / "KaMS-Club-Bot"
LOG_FILE = BOT_DIR / "bot.log"
ENV_FILE = BOT_DIR / ".env"
STATE_FILE = Path.home() / ".config/KaMS-Club-Bot/notify_state.json"
API = "https://discord.com/api/v10"
# Discord expects a bot's requests to name it this way.
USER_AGENT = "DiscordBot (https://github.com/Username2481632/KaMS-Club-Bot, 1)"
OWNER_ZONE = ZoneInfo("America/Los_Angeles")
MESSAGE_LIMIT = 2000
# How the bot's LOGGING_FORMAT starts a record; any other line continues the record before it.
RECORD_START = re.compile(
    r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} \[(\w+)\] (\S+): (.*)"
)
OWN_LOGGERS = re.compile(r"(casy|miphael)(\.(\d+))?$")
# What Python prints outside the logger when nothing catches an exception, or when the logger
# itself fails.
OUTSIDE_LOGGER = {
    "Traceback (most recent call last):": "Uncaught exception",
    "Exception Group Traceback (most recent call last):": "Uncaught exception",
    "--- Logging error ---": "Logging error",
}
# An exception's own line in a traceback, once an exception group's "| " is taken off.
EXCEPTION_LINE = re.compile(r"[A-Za-z_][\w.]*(: .*)?$")


def discord(method: str, path: str, token: str, body: dict | None = None) -> dict:
    """
    Make a request to Discord's API as the bot.
    :param method:
    :param path:
    :param token:
    :param body:
    :return: The decoded response.
    """
    request = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def read_records(text: str) -> list[dict]:
    """
    Split log text into records, each with its continuation lines.
    :param text:
    :return:
    """
    records: list[dict] = []
    for line in text.splitlines():
        start = RECORD_START.match(line)
        outside = OUTSIDE_LOGGER.get(line.lstrip(" +"))
        if start:
            records.append(
                {
                    "time": start[1],
                    "level": start[2],
                    "logger": start[3],
                    "message": start[4],
                    "continuation": [],
                }
            )
        elif outside and (not records or not is_passed(records[-1])):
            records.append(
                {
                    # Printed without a time: the record before it is the nearest.
                    "time": records[-1]["time"] if records else None,
                    "level": "ERROR",
                    "logger": "python",
                    "message": outside,
                    "continuation": [line],
                }
            )
        elif records:
            records[-1]["continuation"].append(line)
    return records


def is_passed(record: dict) -> bool:
    """
    Whether a record is passed on to the owner.
    :param record:
    :return:
    """
    return record["level"] in ("ERROR", "CRITICAL") or (
        record["level"] == "WARNING" and OWN_LOGGERS.match(record["logger"]) is not None
    )


def exception_of(continuation: list[str]) -> str | None:
    """
    The exception a record's traceback ends in.
    :param continuation:
    :return:
    """
    exception: str | None = None
    for line in continuation:
        if line.startswith("Call stack:"):
            # What follows in a logging error is the failed call, not the exception.
            break
        line = re.sub(r"^ *\| ?", "", line)
        if EXCEPTION_LINE.match(line):
            exception = line
    return exception


def discord_length(text: str) -> int:
    """
    The length of a text as Discord's limit counts it, at the most: in UTF-16 code units.
    :param text:
    :return:
    """
    return len(text.encode("utf-16-le")) // 2


def compose(entries: list[list], record_count: int) -> str:
    """
    Write the message for the entries, within Discord's limit.
    :param entries: Each entry's text, when it was first logged and how often.
    :param record_count:
    :return:
    """
    lines = [f"**{record_count} new in Casy's log on miphael:**"]
    for count, (entry, when, times) in enumerate(entries):
        line = f"`{when}` {entry}" + ("" if times == 1 else f" (×{times})")
        remaining = len(entries) - count - 1
        rest = [f"…and {remaining} more in bot.log"] if remaining else []
        if discord_length("\n".join(lines + [line] + rest)) > MESSAGE_LIMIT:
            if count > 0:
                # The line before left room for exactly this.
                lines.append(f"…and {len(entries) - count} more in bot.log")
                break
            # A record too long for a message on its own is cut short.
            while (
                discord_length("\n".join(lines + [line + "…"] + rest)) > MESSAGE_LIMIT
            ):
                line = line[:-1]
            line += "…"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    size = LOG_FILE.stat().st_size
    today = datetime.datetime.now(OWNER_ZONE).date().isoformat()
    if not STATE_FILE.exists():
        # Start watching from here rather than send the whole history.
        save_state({"place": size, "day": today, "reported": []})
        return
    state = json.loads(STATE_FILE.read_text())
    if state["day"] != today:
        state["day"], state["reported"] = today, []
    place = state["place"]
    if place > size:
        # The log was replaced.
        place = 0
    with open(LOG_FILE, "rb") as log:
        log.seek(place)
        text = log.read(size - place)
    # A line still being written is read next time.
    text = text[: text.rfind(b"\n") + 1]
    records = [
        record
        for record in read_records(text.decode(errors="replace"))
        if is_passed(record)
    ]
    if records:
        token = next(
            line.split("=", 1)[1].strip()
            for line in ENV_FILE.read_text().splitlines()
            if line.startswith("DISCORD_BOT_TOKEN=")
        )
        guild_names: dict[str, str] = {}
        # Each distinct entry, numbers aside: its first text, when it was logged and how often.
        entries: dict[str, list] = {}
        record_count = 0
        for record in records:
            where = record["logger"]
            own = OWN_LOGGERS.match(where)
            if own and own[3]:
                if own[3] not in guild_names:
                    try:
                        guild_names[own[3]] = discord(
                            "GET", f"/guilds/{own[3]}", token
                        )["name"]
                    except OSError:
                        guild_names[own[3]] = own[3]
                where = guild_names[own[3]]
            entry = f"**{record['level']}** {where}: {record['message']}"
            exception = exception_of(record["continuation"])
            if exception:
                entry += f" — `{exception.strip()}`"
            key = re.sub(r"\d+", "#", entry)
            if record["logger"] != "miphael" and key in state["reported"]:
                continue
            if key not in entries:
                entries[key] = [
                    entry,
                    datetime.datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=datetime.UTC)
                    .astimezone(OWNER_ZONE)
                    .strftime("%a %H:%M %Z")
                    if record["time"]
                    else "?",
                    0,
                ]
            entries[key][2] += 1
            record_count += 1
        if entries:
            application = discord("GET", "/oauth2/applications/@me", token)
            owner_id = (
                application["team"]["owner_user_id"]
                if application.get("team")
                else application["owner"]["id"]
            )
            channel = discord(
                "POST", "/users/@me/channels", token, {"recipient_id": owner_id}
            )
            discord(
                "POST",
                f"/channels/{channel['id']}/messages",
                token,
                {
                    "content": compose(list(entries.values()), record_count),
                    "allowed_mentions": {"parse": []},
                },
            )
            state["reported"] += list(entries)
    state["place"] = place + len(text)
    save_state(state)


def save_state(state: dict) -> None:
    """
    Remember how far the log has been read and what was reported today, durably: a state lost
    to a power cut would send records again or stop at a file that does not parse.
    :param state:
    """
    temp_file = STATE_FILE.with_suffix(".tmp")
    with open(temp_file, "w") as file:
        json.dump(state, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_file, STATE_FILE)
    directory = os.open(STATE_FILE.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


if __name__ == "__main__":
    main()
