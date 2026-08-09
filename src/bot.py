"""
This bot is designed to manage the KaMS Club Discord server. It assigns roles to members based on their behavior and the roles they have. It also allows members to vote on other members with a severity ranging from -1 to 1. The bot will
automatically time out members with low respect scores and notify them if they have been timed out for more than a certain amount of time. The bot also assigns the Justice role to the top five members with the highest respect scores.

Due to a Discord limitation, you must restrict the /justice_toolbox command visibility to the Justice role manually. To do this, go to server settings -> integrations -> click on "KaMS Club" (manage).

Generating Discord OAuth2 Link:
- Scopes: applications.commands, bot
- Bot Permissions:
  - General: View Audit Log, Manage Roles, Manage Channels, Ban Members, Change Nickname, Moderate Members
  - Text: Send Messages, Send Messages in Threads, Manage Messages, Read Message History
  - Voice: None required.
"""

import asyncio
import datetime
import fractions
import heapq
import json
import logging
import math
import os
import shutil
import signal
import socket
import sys
import time
import traceback
import typing
from types import FrameType, TracebackType
from typing import AsyncGenerator, Callable, Type

import aiofiles
import aiohttp
import discord
import numpy
from discord.ext import commands, tasks
from dotenv import load_dotenv
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
# Set the timezone to UTC
os.environ["TZ"] = "UTC"
time.tzset()
# Ensure working directory is the same as the script's directory. This is crucial for relative paths to work correctly.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
CONFIG_FILE: str = "../config.json"
DATA_FILE: str = "../data.json"
JUSTICE_COUNT: int = 5
JUSTICE_CHANNEL_NAME: str = "justices"
JUSTICE_CHANNEL_CATEGORY: str = "Information"
RESPECTFUL_ROLE_NAME: str = "Respectful :)"
DISRESPECTFUL_ROLE_NAME: str = "Disrespectful :("
TIMEOUT_THRESHOLD: float = (
    -0.3
)  # If a member's shallow score falls below this value, member gets timed out
TIMEOUT_NOTIFICATION_THRESHOLD: datetime.timedelta = datetime.timedelta(
    minutes=0.5
)  # If a member gets timed out for more than this, member gets notified
TIMEOUT_DURATION_OUTLINE: dict[float, float] = {
    1.0: 0.0,
    0.0: 0.0,
    TIMEOUT_THRESHOLD: TIMEOUT_NOTIFICATION_THRESHOLD.total_seconds() / 60.0,
    -1.0: 20.0,
    -2.0: 300.0,
    -3.0: 10080.0,
    -4.0: 10080.0,
}  # Score: Timeout duration (minutes)
MISSING_ROLE_MESSAGE: Callable[[bool, str], str] = lambda timed_out, server_name: (
    f"Hi there. It seems like you're missing some roles in **{server_name}**, which is why {"you've been temporarily timed out" if not timed_out else 'your disrespect timeout has been put on hold and will stop decreasing'}. No worries, "
    f"though! To {'regain access to the server' if not timed_out else "keep serving your existing timeout until it's done"}, just visit the <id:customize> tab to assign yourself the necessary roles. If you have any "
    f"questions or need assistance, feel free to reach out to a moderator. We're here to help!"
)


def ROLE_RESTORATION_MESSAGE(server_name: str) -> str:
    return f"Thanks for acquiring the necessary roles in **{server_name}**. Your timeout has been removed; welcome back!"


def TIMEOUT_RESUME_MESSAGE(
    server_name: str, remaining_timeout: datetime.timedelta
) -> str:
    hours: float = math.ceil(remaining_timeout.seconds / 360.0) / 10.0
    return f"Your role timeout in **{server_name}** has been removed, but you still have an earlier timeout of {f'{remaining_timeout.days} day{"" if remaining_timeout.days == 1 else "s"}' if remaining_timeout.days > 0 else ''} and {hours} hour{'' if hours == 1 else 's'} to serve."


LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
MISSING_ROLE_TIMEOUT_DURATION: datetime.timedelta = datetime.timedelta(days=2)
JUSTICE_DEEP_SCORE_REQUIREMENT: fractions.Fraction = fractions.Fraction(3, 2)
DAY_CHANGE_TIME: datetime.time = datetime.time(hour=0, minute=0, second=0)
JUSTICE_ROLE_NAME = "Justice"
ERROR_SYMBOL = ":x:"
SUCCESS_SYMBOL = ":white_check_mark:"
ELARA_LOGGER_ID: int = 1274076825009655863
ROLE_TIMEOUT_REASON: str = "Missing required roles."
CREDIBILITY_RATIO: fractions.Fraction = fractions.Fraction(
    1, 2**15
)  # Credibility earned per second of conversation
CREDIBILITY_DECAY: int = 10  # Seconds-worth of credibility lost per day
CREDIBILITY_EARNING_EXCLUSION_CHANNELS: list[int] = [
    1201374063810064484,
    1217615412146077806,
    1263269073538515005,
    1217278514298884176,
]
SEVERITY_DISPLAY_PRECISION: int = 4  # Number of decimal places to display for severity

# Record start time
start_time: float = time.time()

# Load environment variables from .env file
load_dotenv()

# Load the token from an environment variable
TOKEN = os.getenv("DISCORD_BOT_TOKEN")


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
class LoggerConfig:
    """
    Class to represent the configuration of a logger.
    """

    def __init__(self, enabled: bool = False, channel_name: str = "logger") -> None:
        self.enabled = enabled
        self.channel_name = channel_name

    def to_dict(self) -> dict:
        """
        Convert the logger config to a dictionary.
        :return:
        """
        return {"enabled": self.enabled, "channel_name": self.channel_name}

    @classmethod
    def from_dict(cls, param):
        """
        Create a LoggerConfig from a dictionary.
        :param param:
        :return:
        """
        return cls(
            enabled=param.get("enabled", False),
            channel_name=param.get("channel_name", "logger"),
        )


class GuildConfigDictType(typing.TypedDict):
    """
    TypedDict for the guild configuration.
    """

    welcome_dm: str | None
    purge_polls: bool
    logger: LoggerConfig
    required_roles: list[list[int]]


class GuildConfig:
    """
    Class to represent the configuration of a guild.
    """

    def __init__(
        self,
        wd: str = "",
        pp: bool = False,
        lc: LoggerConfig = LoggerConfig(),
        rr: list[list[int]] | None = None,
    ) -> None:
        self.welcome_dm = wd
        self.purge_polls = pp
        self.logger = lc
        self.required_roles = [set(roles) for roles in rr] if rr is not None else []

    def to_dict(self) -> GuildConfigDictType:
        """
        Convert the guild config to a dictionary.
        :return:
        """
        return {
            "welcome_dm": self.welcome_dm,
            "purge_polls": self.purge_polls,
            "logger": self.logger,
            "required_roles": [list(roles) for roles in self.required_roles],
        }

    @classmethod
    def from_dict(cls, param) -> "GuildConfig":
        """
        Create a GuildConfig from a dictionary.
        :param param:
        :return:
        """
        return cls(
            wd=param.get("welcome_dm", ""),
            pp=param.get("purge_polls", False),
            lc=LoggerConfig.from_dict(param.get("logger", {})),
            rr=param.get("required_roles", []),
        )


class MemberEntry:
    """
    Class to represent a member entry in the data file.
    """

    def __init__(
        self,
        shallow_score: fractions.Fraction = fractions.Fraction(0),
        deep_score: fractions.Fraction = fractions.Fraction(0),
        credibility: fractions.Fraction = fractions.Fraction(0),
        opinions: dict[int, fractions.Fraction] | None = None,
        latest_message_time: float = discord.utils.DISCORD_EPOCH / 1000,
        conversation_start_time: float = discord.utils.DISCORD_EPOCH / 1000,
        suspended_timeout: float | None = None,
    ) -> None:
        """
        Initialize the member entry.

        :param shallow_score:
        :param deep_score:
        :param credibility:
        :param opinions:
        :param latest_message_time:
        :param conversation_start_time:
        :param suspended_timeout:
        """
        self.shallow_score = shallow_score
        self.deep_score = deep_score
        self.credibility = credibility
        self.opinions = opinions if opinions is not None else {}
        self.latest_message_time = latest_message_time
        self.conversation_start_time = conversation_start_time
        self.suspended_timeout = suspended_timeout

    @classmethod
    def from_dict(cls, data: dict) -> "MemberEntry":
        """
        Create a MemberEntry from a dictionary.

        :param data:
        :return:
        """
        return cls(
            shallow_score=fractions.Fraction(data.get("shallow_score", "0")),
            deep_score=fractions.Fraction(data.get("deep_score", "0")),
            credibility=fractions.Fraction(data.get("credibility", "0")),
            opinions={
                int(k): fractions.Fraction(v)
                for k, v in data.get("opinions", {}).items()
            },
            latest_message_time=data.get(
                "latest_message_time", discord.utils.DISCORD_EPOCH / 1000
            ),
            conversation_start_time=data.get(
                "conversation_start_time", discord.utils.DISCORD_EPOCH / 1000
            ),
            suspended_timeout=data.get("suspended_timeout"),
        )

    def to_dict(self) -> dict:
        """
        Convert the member entry to a dictionary.

        :return:
        """
        return {
            "shallow_score": str(self.shallow_score),
            "deep_score": str(self.deep_score),
            "credibility": str(self.credibility),
            "opinions": {str(k): str(v) for k, v in self.opinions.items()},
            "latest_message_time": self.latest_message_time,
            "conversation_start_time": self.conversation_start_time,
            "suspended_timeout": self.suspended_timeout,
        }


# Define the type for the data structure
ServerDataType = dict[int, MemberEntry]
FullDataType = dict[int, ServerDataType]

GuildsType = dict[int, GuildConfig]

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix="", intents=intents)
# Configure logging, excluding discord logs
logger = logging.getLogger("casy")
logger.setLevel(logging.INFO)
# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
formatter = logging.Formatter(LOGGING_FORMAT)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class GuildLoggerAdapter(logging.LoggerAdapter):
    def log(self, level, msg, *args, **kwargs):
        if not self.isEnabledFor(level):
            return

        # Pop guild_id from kwargs to prevent it from being passed to the final log call
        guild_id = kwargs.pop("guild_id", None)

        # The dot makes this a child of the base logger, so it inherits its handler
        # and level; any other separator gets a parentless logger and no handler.
        target_name = self.logger.name
        if guild_id is not None:
            target_name = f"{self.logger.name}.{guild_id}"

        # Get the target logger and call its log method
        target_logger = logging.getLogger(target_name)
        target_logger.log(level, msg, *args, **kwargs)


logger = GuildLoggerAdapter(logger, {})

guild_objects: list[discord.Guild] = []
is_initialized = False
# Extract x and y coordinates from the dictionary
x_coords: numpy.ndarray = numpy.array(list(TIMEOUT_DURATION_OUTLINE.keys()))
y_coords: numpy.ndarray = numpy.array(list(TIMEOUT_DURATION_OUTLINE.values()))
# Create a linear interpolation function
linear_interp = interp1d(
    x_coords, y_coords, fill_value="extrapolate"
)  # linear interpolation
# Generate points to plot the function
x_values: numpy.ndarray = numpy.linspace(min(x_coords), max(x_coords), 500)
y_values: numpy.ndarray = linear_interp(x_values)
shutdown_event = asyncio.Event()
# Initialize a lock for thread-safe file access
data_lock = asyncio.Lock()

# Global guild configurations - will be initialized during bot startup
GUILDS: GuildsType = {}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


# Function to evaluate the linear interpolation at any given x
def calculate_timeout(x: fractions.Fraction) -> float:
    """
    Calculate the timeout duration based on the shallow score.
    :param x:
    :return:
    """
    return float(linear_interp(float(x)))


async def update_bot_nickname(guild: discord.Guild) -> None:
    """
    Update bot's nickname to match server name if not already correct
    :param guild:
    :return:
    """
    expected_nick = f"{guild.name} Bot"
    if guild.me.nick == expected_nick:
        return

    try:
        await guild.me.edit(nick=expected_nick)
        logger.info(f"Updated nickname to '{expected_nick}'", guild_id=guild.id)
    except discord.Forbidden:
        logger.error("Missing permissions to set nickname", guild_id=guild.id)
    except Exception as e:
        logger.error(f"Error setting nickname: {e}", guild_id=guild.id)


async def load_data() -> FullDataType:
    """
    Load data from the JSON file asynchronously.
    :return: The data dictionary.
    """
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                return {
                    int(key1): {
                        int(key2): MemberEntry.from_dict(value)
                        for key2, value in subdict.items()
                    }
                    for key1, subdict in json.load(file).items()
                }
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading data: {e}")
    return {}


async def save_data(data: FullDataType, output_file: str = DATA_FILE) -> None:
    """
    Save data to the JSON file asynchronously.
    :param data: The data dictionary to save.
    :param output_file: Path to the output JSON file.
    """
    json_data = json.dumps(
        {
            str(key1): {str(key2): value.to_dict() for key2, value in subdict.items()}
            for key1, subdict in data.items()
        },
        indent=2,
    )
    # Written alongside the target and renamed over it, so an interrupted write leaves
    # the previous file rather than a truncated one that would load as no data at all.
    temp_file: str = f"{output_file}.tmp"
    async with aiofiles.open(temp_file, "w", encoding="utf-8") as file:
        await file.write(json_data)
    os.replace(temp_file, output_file)


def format_severity(severity: fractions.Fraction) -> str:
    """
    Format the credibility value for display.
    :param severity:
    :return:
    """
    min_value: float = 10**-SEVERITY_DISPLAY_PRECISION
    if 0 < severity < min_value:
        return f"<{min_value}"
    elif -min_value < severity < 0:
        return f"-(<{min_value})"
    elif severity == 0:
        return "0"
    else:
        return (
            str(round(float(severity), SEVERITY_DISPLAY_PRECISION))
            .rstrip("0")
            .rstrip(".")
        )


async def get_justice_role(guild: discord.Guild) -> discord.Role:
    """
    Returns the Justice role in the given guild, creating one if it doesn't exist.

    :param guild: The guild to get the role from.
    :return: The Justice role.
    """
    justice_role: discord.Role | None = discord.utils.get(
        guild.roles, name=JUSTICE_ROLE_NAME
    )
    if not justice_role:
        logger.warning("Justice role missing", guild_id=guild.id)
        justice_role = await guild.create_role(
            name=JUSTICE_ROLE_NAME,
            hoist=True,
            reason="Created by bot to keep track of justices.",
        )
        logger.info(
            f"Created blank justice role (id={justice_role.id})—IT IS ADVISABLE TO CUSTOMIZE IT WITH PERMISSIONS AND DISPLAY OPTIONS (this will only be shown once).",
            guild_id=guild.id,
        )
    return justice_role


# -----------------------------------------------------------------------------
# Main logic functions
# -----------------------------------------------------------------------------


async def set_respect_role(
    guild: discord.Guild, member: discord.Member, score: fractions.Fraction
) -> None:
    """
    Set the respect role based on the score.
    :param guild:
    :param member:
    :param score:
    :return:
    """
    disrespectful_role: discord.Role | None = discord.utils.get(
        guild.roles, name=DISRESPECTFUL_ROLE_NAME
    )
    respectful_role: discord.Role | None = discord.utils.get(
        guild.roles, name=RESPECTFUL_ROLE_NAME
    )

    if disrespectful_role is None or respectful_role is None:
        both_missing: bool = disrespectful_role is None and respectful_role is None
        logger.warning(
            f"The {f"'{DISRESPECTFUL_ROLE_NAME}' " if disrespectful_role is None else ''}{'and ' if both_missing else ''}{f"'{RESPECTFUL_ROLE_NAME}' " if respectful_role is None else ''}role{'s' if both_missing else ''} do{'es' if respectful_role is not None or disrespectful_role is not None else ''} not exist. Creating {'them' if both_missing else 'it'}.",
            guild_id=guild.id,
        )
        i = False
        while True:
            if (disrespectful_role if i else respectful_role) is None:
                try:
                    created_role = await guild.create_role(
                        name=DISRESPECTFUL_ROLE_NAME if i else RESPECTFUL_ROLE_NAME,
                        reason="Created by bot for voting system.",
                    )
                    # Update the appropriate variable
                    if i:
                        disrespectful_role = created_role
                    else:
                        respectful_role = created_role
                except discord.Forbidden:
                    logger.error(
                        f"Missing permissions to create '{DISRESPECTFUL_ROLE_NAME if i else RESPECTFUL_ROLE_NAME}' role",
                        guild_id=guild.id,
                    )
            if i:
                break
            i = True

    if score >= 0.0:
        if disrespectful_role in member.roles:
            await member.remove_roles(
                disrespectful_role, reason=f"Respect score of {score} is positive."
            )
        if respectful_role not in member.roles:
            await member.add_roles(
                respectful_role, reason=f"Respect score of {score} is positive."
            )
            logger.info(
                f"{member.display_name} has been upgraded to '{RESPECTFUL_ROLE_NAME}'.",
                guild_id=guild.id,
            )
    elif disrespectful_role not in member.roles and respectful_role not in member.roles:
        await member.add_roles(disrespectful_role, reason="Bad respect score.")
        logger.info(
            f"{member.display_name} has been assigned '{DISRESPECTFUL_ROLE_NAME}' because their roles were missing and their respect score is negative.",
            guild_id=guild.id,
        )
    elif score < min(-1.0, -0.01 * sum(not memb.bot for memb in guild.members)):
        if respectful_role in member.roles:
            await member.remove_roles(
                respectful_role, reason=f"Respect score of {score} is unacceptably bad."
            )
            if disrespectful_role not in member.roles:
                await member.add_roles(disrespectful_role)
                logger.info(
                    f"{member.display_name} has been downgraded to '{DISRESPECTFUL_ROLE_NAME}'.",
                    guild_id=guild.id,
                )


# -----------------------------------------------------------------------------
# Event listeners
# -----------------------------------------------------------------------------


@bot.event
async def on_message(message: discord.Message, override: bool = False) -> None:
    """

    :param override:
    :param message:
    """
    if not is_initialized and not override or message.guild is None:
        return
    # update the user's [latest_message_time] and [conversation_start_time] in the data file
    author_id: int = message.author.id
    if message.author == bot.user and message.mentions:
        author_id = message.mentions[0].id

    async with data_lock:
        data: FullDataType = await load_data()
        # Ensure the guild exists in the data
        if message.guild.id not in data:
            data[message.guild.id] = {}
        if author_id not in data[message.guild.id]:
            await _on_member_join_impl(message.author, data, message.guild)

        # Get the timestamp of the message (use edited_at if available, else use created_at)
        message_timestamp: float = (
            message.edited_at.timestamp()
            if message.edited_at
            else message.created_at.timestamp()
        )

        # Check if the difference in time is greater than 300 seconds (5 minutes)
        if (
            message_timestamp - data[message.guild.id][author_id].latest_message_time
            > 300
        ):
            # Update credibility based on the time difference and reset conversation start time
            data[message.guild.id][author_id].credibility += (
                fractions.Fraction(
                    data[message.guild.id][author_id].latest_message_time
                    - data[message.guild.id][author_id].conversation_start_time
                )
                * CREDIBILITY_RATIO
            )
            data[message.guild.id][
                author_id
            ].conversation_start_time = message_timestamp

        data[message.guild.id][author_id].latest_message_time = message_timestamp
        await save_data(data)


# -----------------------------------------------------------------------------
# Justice toolbox
# -----------------------------------------------------------------------------


class JusticeToolboxView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Set Slowmode", style=discord.ButtonStyle.primary)
    async def set_slowmode(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ):
        """
        :param interaction:
        :param _button:
        """
        modal = SetSlowmodeModal()
        # noinspection PyUnresolvedReferences
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Request Ban/Unban", style=discord.ButtonStyle.danger)
    async def request_ban(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ):
        """

        :param interaction:
        :param _button:
        """
        select = BanUnbanView()
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message(
            "Select whether you would like request to ban or to unban a user.",
            view=select,
            ephemeral=True,
        )


class SetSlowmodeModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="Set Slowmode for the Current Channel")
        self.length = discord.ui.TextInput(label="Length, seconds", required=True)
        self.reset_time = discord.ui.TextInput(
            label="Reset Time, minutes (optional)", required=False
        )
        self.add_item(self.length)
        self.add_item(self.reset_time)

    async def on_submit(self, interaction: discord.Interaction):
        # Validate input for length
        try:
            length = int(self.length.value)
            if length < 0:
                raise ValueError("Slowmode must be a non-negative integer.")
        except ValueError:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                content=f"{ERROR_SYMBOL} Invalid slowmode length. Slowmode must be a non-negative integer."
            )
            return

        # Optional reset time validation
        reset_time: float | None = None
        if self.reset_time.value:
            try:
                reset_time = float(self.reset_time.value)
                if reset_time <= 0:
                    raise ValueError("Reset time must be a positive number.")
            except ValueError:
                # noinspection PyUnresolvedReferences
                await interaction.response.edit_message(
                    content=f"{ERROR_SYMBOL} Invalid reset time. Please enter a positive number."
                )
                return

        if length < 0:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                content=f"{ERROR_SYMBOL} Slowmode must be a non-negative integer."
            )
            return
        try:
            await interaction.channel.edit(
                slowmode_delay=length,
                reason=f'Set by user {interaction.user.id} ("{interaction.user.display_name}") via Justice Toolbox',
            )
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                content=f"{SUCCESS_SYMBOL} Slowmode for {interaction.channel.mention} has been set to {length} second{'s' if length != 1 else ''} {f'with a reset time of {reset_time} minutes' if reset_time is not None else ''}."
            )
            if reset_time is not None:
                await asyncio.sleep(reset_time * 60.0)
                await interaction.channel.edit(
                    slowmode_delay=0,
                    reason=f"Reset from command by user {interaction.user.id} via Justice Toolbox",
                )
        except discord.errors.Forbidden:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                f"{ERROR_SYMBOL} I do not have permission to set slowmode in this channel."
            )


BanRequestType = dict[str, bool | str]  # {"request": bool, "reason": str}
BanRequestsType = dict[
    int, dict[int, dict[int, BanRequestType]]
]  # {server_id: {user_id: {requester_id: BanRequest}}}


class RequestBanModal(discord.ui.Modal):
    def __init__(self, ban_bool: bool, **kwargs):
        title = "Request Ban" if ban_bool else "Request Unban"
        super().__init__(title=title, **kwargs)
        self.ban_bool = ban_bool
        self.target: discord.ui.TextInput = discord.ui.TextInput(
            label=f"User ID to {'Ban' if self.ban_bool else 'Unban'}",
            required=True,
            placeholder="012345678910111213",
            style=discord.TextStyle.short,
        )
        self.reason = discord.ui.TextInput(
            label=f"Reason for {'Ban' if self.ban_bool else 'Unban'}",
            style=discord.TextStyle.paragraph,
            required=True,
            min_length=60,
        )

        # Add the text inputs to the modal
        self.add_item(self.target)
        self.add_item(self.reason)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            target_object: discord.User = await bot.fetch_user(int(self.target.value))
        except (ValueError, discord.errors.NotFound, discord.errors.HTTPException):
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                content=f"{ERROR_SYMBOL} Invalid user ID. Please enter a valid user ID."
            )
            return

        reason = self.reason.value
        # Open file ban_requests.json, create it if it doesn't exist
        ban_requests: BanRequestsType = read_ban_requests()
        if target_object.id not in ban_requests[interaction.guild.id]:
            ban_requests[interaction.guild.id][target_object.id] = {}
        existing_request: dict[str, bool | str] | None = ban_requests[
            interaction.guild.id
        ][target_object.id].get(interaction.user.id)
        request_changed: bool = (
            existing_request["request"] != self.ban_bool if existing_request else False
        )
        ban_requests[interaction.guild.id][target_object.id][interaction.user.id] = {
            "request": self.ban_bool,
            "reason": reason,
        }
        save_ban_requests(ban_requests)
        if self.ban_bool:
            # If all current justices have requested a ban, ban the user
            justice_ids: list[int] = await get_justice_ids(interaction.guild)
            if len(justice_ids) == JUSTICE_COUNT and all(
                justice_id in ban_requests[interaction.guild.id][target_object.id]
                and ban_requests[interaction.guild.id][target_object.id][justice_id]
                for justice_id in justice_ids
            ):
                # Ban the user
                try:
                    await interaction.guild.ban(
                        target_object, reason="Requested by justices."
                    )
                except discord.errors.Forbidden:
                    # noinspection PyUnresolvedReferences
                    await interaction.response.edit_message(
                        f"{ERROR_SYMBOL} Sorry, I am unable to ban this user."
                    )
        else:
            # If 2/3 of current justices have requested an unban, unban the user
            justice_ids: list[int] = await get_justice_ids(interaction.guild)
            if sum(
                1
                for justice_id in justice_ids
                if justice_id in ban_requests[interaction.guild.id][target_object.id]
                and not ban_requests[interaction.guild.id][target_object.id][justice_id]
            ) >= 2 * len(justice_ids) / 3 and any(
                ban.user.id == target_object.id
                for ban in [ban async for ban in interaction.guild.bans()]
            ):
                # Unban the user
                await interaction.guild.unban(
                    target_object, reason="Requested by justices."
                )
        # noinspection PyUnresolvedReferences
        await interaction.response.edit_message(
            content=f'{SUCCESS_SYMBOL} {f"{'Unb' if not self.ban_bool else 'B'}an r" if not request_changed else "R"}equest {"submitted" if not existing_request else "updated" + (f" from **{'ban' if existing_request['request'] else 'unban'}** to **{'ban' if self.ban_bool else 'unban'}**" if request_changed else "")} for user {self.target.value} ("{target_object.display_name}").'
        )


class BanUnbanSelect(discord.ui.Select):
    def __init__(self):
        # To make it a multi-select dropdown, add the parameter max_values=2
        options = [
            discord.SelectOption(label="Ban", value="ban"),
            discord.SelectOption(label="Unban", value="unban"),
            discord.SelectOption(label="Cancel Prior Request", value="cancel"),
        ]
        super().__init__(placeholder="Select an action", options=options)

    async def callback(self, interaction: discord.Interaction):
        # Save the user's choice in the interaction or context
        choice = self.values[0]
        if choice == "cancel":
            ban_requests: BanRequestsType = read_ban_requests()
            user_requests: list[tuple[int, str, bool]] = []
            for target_id in ban_requests[interaction.guild.id]:
                if interaction.user.id in ban_requests[interaction.guild.id][target_id]:
                    target_object: discord.User = await bot.fetch_user(target_id)
                    user_requests.append(
                        (
                            target_id,
                            target_object.display_name,
                            ban_requests[interaction.guild.id][target_id][
                                interaction.user.id
                            ]["request"],
                        )
                    )
            if not user_requests:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(
                    f"{ERROR_SYMBOL} You have no ban requests to cancel.",
                    ephemeral=True,
                    delete_after=15,
                )
            else:
                select = BanRequestCancelSelect(requests=user_requests)
                view: discord.ui.View = discord.ui.View()
                view.add_item(select)
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(
                    "Select a ban request to cancel.",
                    view=view,
                    ephemeral=True,
                    delete_after=60,
                )
            return
        # Pass the choice to the modal
        modal = RequestBanModal(ban_bool=choice == "ban")
        # noinspection PyUnresolvedReferences
        await interaction.response.send_modal(modal)


class BanRequestCancelSelect(discord.ui.Select):
    def __init__(self, requests: list[tuple[int, str, bool]]):
        # Ban/Unban [User id] ("display name")
        options = [
            discord.SelectOption(
                label=f'{"Ban" if request[2] else "Unban"} {request[0]} ("{request[1]}")',
                value=str(request[0]),
            )
            for request in requests
        ]
        super().__init__(placeholder="Select a ban request to cancel", options=options)
        self.requests = requests

    async def callback(self, interaction: discord.Interaction):
        # Delete the ban request from the file
        target_id = int(self.values[0])
        ban_requests: BanRequestsType = read_ban_requests()
        for requester_id in ban_requests[interaction.guild.id][target_id]:
            if requester_id == interaction.user.id:
                ban_requests[interaction.guild.id][target_id].pop(requester_id)
                save_ban_requests(ban_requests)
                # noinspection PyUnresolvedReferences
                # Find the user id in the self.requests list and get the display name
                await interaction.response.send_message(
                    f'{SUCCESS_SYMBOL} Ban request for user {target_id} ("{next(request[1] for request in self.requests if request[0] == target_id)}") has been cancelled.',
                    ephemeral=True,
                )
                break


class BanUnbanView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(BanUnbanSelect())


def read_ban_requests() -> BanRequestsType:
    """
    Read the ban requests from the JSON file.
    :return:
    """
    if os.path.exists("../ban_requests.json"):
        with open("../ban_requests.json") as file:
            return {
                int(guild_key): {
                    int(target_key): {
                        int(user_key): value for user_key, value in subdict.items()
                    }
                    for target_key, subdict in guild_dict.items()
                }
                for guild_key, guild_dict in json.load(file).items()
            }
    return {}


def save_ban_requests(ban_requests: BanRequestsType) -> None:
    """
    Save the ban requests to the JSON file.
    :param ban_requests:
    """
    with open("../ban_requests.json", "w") as file:
        json.dump(
            {
                str(guild_key): {
                    str(target_key): {
                        str(user_key): value for user_key, value in subdict.items()
                    }
                    for target_key, subdict in guild_dict.items()
                }
                for guild_key, guild_dict in ban_requests.items()
            },
            file,
            indent=2,
        )


async def dm_member(
    member: discord.Member | discord.User, message: str, guild_id: int
) -> None:
    """
    Send a direct message to a member, creating a DM channel if necessary.
    :param member:
    :param message:
    :param guild_id:
    """

    try:
        await member.send(message)
    except discord.errors.Forbidden:
        logger.error(
            f'Forbidden to send message to "{member.display_name}" (id={member.id}).',
            guild_id=guild_id,
        )


async def message_generator(
    channel, after: datetime.datetime
) -> AsyncGenerator[discord.Message, None]:
    """
    Async generator that yields messages from a channel in chronological order.
    """
    try:
        async for message in channel.history(
            after=after, oldest_first=True, limit=None
        ):
            if shutdown_event.is_set():
                logger.info(
                    f"Shutdown requested. Aborting message gathering in {channel.name}.",
                    guild_id=channel.guild.id,
                )
                return
            yield message
    except discord.Forbidden:
        logger.info(
            f"No permission to read history in channel {channel.name} ({channel.id})",
            guild_id=channel.guild.id,
        )
    except discord.HTTPException as e:
        logger.error(
            f"Failed to fetch messages from channel {channel.name} ({channel.id}): {e}",
            guild_id=channel.guild.id,
        )


def collect_generators(
    channel: discord.abc.GuildChannel,
    after_time: datetime.datetime,
    processed: set[int],
) -> list[AsyncGenerator[discord.Message, None]]:
    """
    Collect async message generators while tracking processed channels
    """
    generators = []

    # Skip excluded channels and already processed channels
    if channel.id in CREDIBILITY_EARNING_EXCLUSION_CHANNELS or channel.id in processed:
        return generators

    processed.add(channel.id)

    if isinstance(channel, discord.CategoryChannel):
        # Process category children
        for subchannel in channel.channels:
            generators.extend(collect_generators(subchannel, after_time, processed))
    elif isinstance(channel, (discord.TextChannel, discord.VoiceChannel)):
        # Process text/voice channel and its threads
        generators.append(message_generator(channel, after_time))
        if isinstance(channel, discord.TextChannel):
            for thread in channel.threads:
                if thread.id not in processed:
                    generators.append(message_generator(thread, after_time))
                    processed.add(thread.id)
    elif isinstance(channel, discord.ForumChannel):
        # Process forum channel threads
        for thread in channel.threads:
            if thread.id not in processed:
                generators.append(message_generator(thread, after_time))
                processed.add(thread.id)

    return generators


async def process_messages_in_order(
    generators: list[AsyncGenerator[discord.Message, None]],
) -> None:
    """
    Process messages from multiple generators in chronological order using a priority queue.
    """
    heap = []
    # Initialize heap with first message from each generator
    for gen in generators:
        try:
            msg = await gen.__anext__()
            heapq.heappush(heap, (msg.created_at.timestamp(), id(gen), gen, msg))
        except StopAsyncIteration:
            continue
        except Exception as e:
            logger.error(f"Error retrieving message: {e}", guild_id=None)
            continue

    while heap:
        if shutdown_event.is_set():
            logger.info(
                "Shutdown requested. Halting missed-message processing.", guild_id=None
            )
            return

        created_at_ts, gen_id, gen, msg = heapq.heappop(heap)
        await on_message(msg, True)

        try:
            next_msg = await gen.__anext__()
            heapq.heappush(
                heap, (next_msg.created_at.timestamp(), id(gen), gen, next_msg)
            )
        except StopAsyncIteration:
            pass
        except Exception as e:
            logger.error(f"Error retrieving next message: {e}", guild_id=None)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


@bot.event
async def on_ready() -> None:
    """
    Event that runs when the bot is ready, syncing the commands and starting the day_change loop.
    """
    global is_initialized, guild_objects, GUILDS
    if is_initialized:
        return

    # Config file check and creation
    if not os.path.exists(CONFIG_FILE):
        json.dump(
            {str(guild.id): GuildConfig().to_dict() for guild in bot.guilds},
            open(CONFIG_FILE, "w"),
            indent=2,
        )
        logger.info(
            "Config file not found. Created a default one with all current guilds.",
            guild_id=None,
        )
    else:
        try:
            with open(CONFIG_FILE) as config_file:
                config_data: dict = json.load(config_file)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse config file {CONFIG_FILE}: {str(e)}", guild_id=None
            )
            await shutdown()
            return

    # Get default configs for validation
    default_guild_config: dict = GuildConfig().to_dict()
    expected_guild_keys: set[str] = set(default_guild_config.keys())
    default_logger_config: dict = LoggerConfig().to_dict()
    expected_logger_keys: set[str] = set(default_logger_config.keys())

    guilds: GuildsType = {}

    for g_id_str, g_cfg in config_data.items():
        # Validate guild ID format
        try:
            g_id: int = int(g_id_str)
        except ValueError:
            logger.error(
                f"Invalid guild ID '{g_id_str}' - must be integer. Skipping entry.",
                guild_id=None,
            )
            continue

        # Get guild info for logging
        guild: discord.Guild | None = bot.get_guild(g_id)
        if guild is None:
            logger.error(
                f"Configured guild id={g_id} not found - bot not in server. Skipping entry.",
                guild_id=None,
            )
            continue
        guild_label: str = f"{guild.name}"

        # Validate top-level keys
        present_guild_keys: set[str] = set(g_cfg.keys())

        # Check for missing keys
        for missing_key in expected_guild_keys - present_guild_keys:
            logger.warning(
                f"{guild_label}: Missing config key '{missing_key}' - using default value.",
                guild_id=g_id,
            )

        # Check for unknown top-level keys
        for unknown_key in present_guild_keys - expected_guild_keys:
            logger.warning(
                f"{guild_label}: Unknown config key '{unknown_key}'", guild_id=g_id
            )

        # Validate logger config
        logger_cfg: dict = g_cfg.get("logger", {})
        present_logger_keys: set[str] = set(logger_cfg.keys())

        # Check for missing logger keys
        for missing_key in expected_logger_keys - present_logger_keys:
            logger.warning(
                f"{guild_label} Logger: Missing config key '{missing_key}' - using default value.",
                guild_id=g_id,
            )

        # Check for unknown logger keys
        for unknown_key in present_logger_keys - expected_logger_keys:
            logger.warning(
                f"{guild_label} Logger: Unknown key '{unknown_key}'", guild_id=g_id
            )

        # Validate required_roles structure
        required_roles: list = g_cfg.get("required_roles", [])
        if not isinstance(required_roles, list):
            logger.error(
                f"{guild_label}: Invalid required_roles format, must be list of role lists. Assuming no requirements.",
                guild_id=g_id,
            )
            required_roles = []

        # Validate individual role groups
        valid_roles: list[list[int]] = []
        for role_group in required_roles:
            if not isinstance(role_group, list):
                logger.error(
                    f'{guild_label}: Invalid required_roles group format "{role_group}", must be list of role IDs. Skipping group.',
                    guild_id=g_id,
                )
                continue
            valid_group: list[int] = []
            for role_id in role_group:
                if not isinstance(role_id, int):
                    logger.error(
                        f"{guild_label}: Non-integer role ID {role_id} found. Skipping group.",
                        guild_id=g_id,
                    )
                    valid_group = []
                    break
                if not guild.get_role(role_id):
                    logger.error(
                        f"{guild_label}: Role ID {role_id} not found in guild. Skipping group.",
                        guild_id=g_id,
                    )
                    valid_group = []
                    break
                valid_group.append(role_id)
            if valid_group:
                valid_roles.append(valid_group)

        # Create validated config
        try:
            guilds[g_id] = GuildConfig.from_dict(
                {
                    **default_guild_config,  # Start with defaults
                    **g_cfg,  # Override with config values
                    "required_roles": valid_roles,
                }
            )
        except Exception as e:
            logger.error(
                f"{guild_label}: Failed to create config - {str(e)}.", guild_id=g_id
            )

    # Final guild verification
    for g_id in list(guilds.keys()):
        guild: discord.Guild | None = bot.get_guild(g_id)
        if not guild:
            logger.error(
                f"Configured guild id={g_id} not found - bot not in server. Removing from config.",
                guild_id=None,
            )
            del guilds[g_id]

    if not guilds:
        logger.error("No valid guild configurations found.", guild_id=None)
        await shutdown()
        return

    # Atomically set GUILDS after all validation succeeds
    # Mutate existing GUILDS so a missing name raises instead of silently creating a local.
    GUILDS.clear()
    GUILDS.update(guilds)

    # Initialize guild objects
    guild_objects = [bot.get_guild(g_id) for g_id in GUILDS.keys()]

    logger.info("Verifying bot nicknames...", guild_id=None)
    for gld in guild_objects:
        await update_bot_nickname(gld)

    @bot.tree.command(
        name="justice_toolbox",
        description="Access the Justice Toolbox.",
        guilds=guild_objects,
    )
    async def slash_justice_toolbox(interaction: discord.Interaction) -> None:
        """
        Access the Justice Toolbox.
        :param interaction:
        """
        justice_role: discord.Role = await get_justice_role(interaction.guild)
        if justice_role not in interaction.user.roles:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message(
                "You must be a Justice to access the Justice Toolbox.", ephemeral=True
            )
            return
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message(
            "Justice Toolbox", view=JusticeToolboxView(), ephemeral=True
        )

    @bot.tree.command(
        name="my_opinions",
        description="View your opinions, constructed from your votes.",
        guilds=guild_objects,
    )
    async def slash_my_opinions(interaction: discord.Interaction) -> None:
        """
        Output a table of percentages, adding to <= 1
        """
        output = ""
        async with data_lock:
            data: FullDataType = await load_data()
            if interaction.user.id not in data[interaction.guild.id]:
                await _on_member_join_impl(interaction.user, data, interaction.guild)

            if len(data[interaction.guild.id][interaction.user.id].opinions) == 0:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(
                    "You have not voted on anyone yet.", ephemeral=True
                )
                return
            for target_id, severity in data[interaction.guild.id][
                interaction.user.id
            ].opinions.items():
                target: discord.User = await bot.fetch_user(target_id)
                output += f"**{target.display_name}**: {format_severity(severity)}\n"
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message(output, ephemeral=True)

    @bot.tree.command(
        name="vote",
        description="Vote for a user with a severity ranging from -1 to 1. See The Rules for more information.",
        guilds=guild_objects,
    )
    async def slash_vote(
        interaction: discord.Interaction,
        target: discord.User,
        severity: float,
        reason: str,
        hidden: bool,
    ) -> None:
        """
        Vote for a user with a severity ranging from -1 to 1.
        :param hidden:
        :param reason:
        :param interaction:
        :param target:
        :param severity:
        :return:
        """
        fraction_severity: fractions.Fraction = fractions.Fraction(severity)
        if fraction_severity == 0:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message(
                "You cannot vote with a severity of 0.", ephemeral=True
            )
            return

        # Check if bot has permission to send messages for non-hidden votes
        if (
            not hidden
            and not interaction.channel.permissions_for(
                interaction.guild.me
            ).send_messages
        ):
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message(
                "Sorry, I'm missing permissions to send messages in this channel. Please contact a server admin or try voting in another channel.",
                ephemeral=True,
            )
            return
        async with data_lock:
            data: FullDataType = await load_data()
            if interaction.user.id not in data[interaction.guild.id]:
                await _on_member_join_impl(interaction.user, data, interaction.guild)
            target_member: discord.Member | None = interaction.guild.get_member(
                target.id
            )
            # If the target is not in the server, still process the vote but just don't take immediate action
            if (
                target.id not in data[interaction.guild.id]
                and target_member is not None
            ):
                await _on_member_join_impl(target_member, data, interaction.guild)
            if fraction_severity < -1 or fraction_severity > 1:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(
                    "Invalid severity value. Please use a value between -1 and 1.",
                    ephemeral=True,
                )
                logger.info(
                    f"Invalid severity value for {interaction.user.display_name} to vote for {target.display_name} with severity {fraction_severity}.",
                    guild_id=interaction.guild.id,
                )
                return
            data[interaction.guild.id][interaction.user.id].opinions[target.id] = (
                (
                    data[interaction.guild.id][interaction.user.id].opinions[target.id]
                    + fraction_severity
                )
                if target.id in data[interaction.guild.id][interaction.user.id].opinions
                else fraction_severity
            )

            # And adjust the rest of the user's opinions to make sure their absolute sum is less than or equal to 1
            adjust_factor: fractions.Fraction = fractions.Fraction(
                1,
                max(
                    1,
                    sum(
                        abs(value)
                        for value in data[interaction.guild.id][
                            interaction.user.id
                        ].opinions.values()
                    ),
                ),
            )
            fraction_severity *= adjust_factor
            for key in data[interaction.guild.id][interaction.user.id].opinions:
                data[interaction.guild.id][interaction.user.id].opinions[key] *= (
                    adjust_factor
                )
            assert (
                sum(
                    map(
                        abs,
                        data[interaction.guild.id][
                            interaction.user.id
                        ].opinions.values(),
                    )
                )
                <= 1
            )

            data[interaction.guild.id][target.id].shallow_score = data[
                interaction.guild.id
            ][target.id].shallow_score + fraction_severity * max(
                data[interaction.guild.id][interaction.user.id].credibility,
                fractions.Fraction(1, 100),
            )
            if target_member is not None:
                await set_respect_role(
                    interaction.guild,
                    target_member,
                    data[interaction.guild.id][target.id].shallow_score
                    + data[interaction.guild.id][target.id].deep_score,
                )
                if data[interaction.guild.id][target.id].shallow_score < (
                    TIMEOUT_THRESHOLD + 1.0
                ):
                    # Timeout procedure
                    timeout_minutes = calculate_timeout(
                        data[interaction.guild.id][target.id].shallow_score
                        + min(
                            data[interaction.guild.id][target.id].deep_score,
                            fractions.Fraction(1, 2),
                        )
                    )
                    old_duration: datetime.timedelta = datetime.timedelta()
                    if (
                        target_member.timed_out_until is not None
                        and (target_member.timed_out_until - discord.utils.utcnow())
                        > old_duration
                    ):
                        old_duration = (
                            target_member.timed_out_until - discord.utils.utcnow()
                        )
                    new_duration: datetime.timedelta = datetime.timedelta(
                        minutes=timeout_minutes
                    )
                    if (
                        fraction_severity < 0 or new_duration < old_duration
                    ) and new_duration != old_duration:
                        if (
                            data[interaction.guild.id][
                                target_member.id
                            ].suspended_timeout
                            is not None
                        ):
                            data[interaction.guild.id][
                                target_member.id
                            ].suspended_timeout = new_duration.total_seconds()
                        else:
                            await _smart_timeout(
                                target_member,
                                new_duration,
                                f"Voted {fraction_severity} by a member.",
                                f"You have been timed out for {timeout_minutes} minutes due to your low respect score. Please take this time to reflect on your behavior. If you have any questions, feel free to reach out to a moderator."
                                if old_duration
                                < TIMEOUT_NOTIFICATION_THRESHOLD
                                < new_duration
                                else None,
                            )
            await save_data(data)

            formatted_severity: str = format_severity(fraction_severity)
            if not hidden:
                # Send a message publicly
                public_message: str = f"{interaction.user.mention} has {'up' if fraction_severity > 0 else 'down'}voted {target.mention} with severity {formatted_severity}. Reason: {reason}"  # If changing this line, also update on_message.
                await interaction.channel.send(public_message)
            try:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(
                    f"Vote successful! Your opinion on {target.display_name} is now {format_severity(data[interaction.guild.id][interaction.user.id].opinions[target.id])}",
                    ephemeral=True,
                    delete_after=15,
                )
            except discord.errors.NotFound:
                logger.error(
                    f'Interaction not found to send vote confirmation to "{interaction.user.display_name}". Processing may have taken too long. Proceeding to send a DM.',
                    guild_id=interaction.guild.id,
                )
                await dm_member(
                    interaction.user,
                    f"With apologies for the delay, your vote for {target.display_name} with severity {formatted_severity} has been successfully processed. Your opinion on {target.display_name} is now "
                    f"{data[interaction.guild.id][interaction.user.id].opinions[target.id]}.",
                    interaction.guild.id,
                )

    async with data_lock:
        logger.info("Bot is ready, starting to sync commands...", guild_id=None)
        commands_synced: list[discord.app_commands.AppCommand] = []
        for gld in guild_objects:
            commands_synced.extend(await bot.tree.sync(guild=gld))

        assert not bot.tree.get_commands()
        assert len(commands_synced) == sum(
            len(bot.tree.get_commands(guild=g)) for g in guild_objects
        )
        logger.info("Slash commands synced!", guild_id=None)
        day_change.start()
        logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})", guild_id=None)
        logger.info("Catching up on missed messages...", guild_id=None)

        dta = await load_data()
        # Add missing guilds to the data file
        was_modified: bool = False
        for g in guild_objects:
            if g.id not in dta:
                dta[g.id] = {}
                was_modified = True
        if was_modified:
            await save_data(dta)
        after_time = datetime.datetime.fromtimestamp(
            max(
                [discord.utils.DISCORD_EPOCH / 1000]
                + [
                    member_entry.latest_message_time
                    for g in guild_objects
                    for member_entry in dta[g.id].values()
                ]
            )
        )

    # Collect all message generators with duplicate prevention
    processed_channels = set()
    generators = []
    for g in guild_objects:
        for channel in g.channels:
            generators.extend(
                collect_generators(channel, after_time, processed_channels)
            )

        # Also check for any threads that might not be in channel.threads
        # (Discord.py sometimes doesn't load all threads immediately)
        for thread in g.threads:
            if (
                thread.id not in processed_channels
                and thread.id not in CREDIBILITY_EARNING_EXCLUSION_CHANNELS
            ):
                generators.append(message_generator(thread, after_time))
                processed_channels.add(thread.id)

    # Process messages in order
    await process_messages_in_order(generators)

    is_initialized = True
    logger.info("Initialization complete.", guild_id=None)


async def get_justice_ids(guild: discord.Guild) -> list[int]:
    """
    Get the justice ids from the justices channel.

    :return:
    """
    # Fetch justice ids from the justices channel
    justice_channel_category: discord.CategoryChannel | None = discord.utils.get(
        guild.categories, name=JUSTICE_CHANNEL_CATEGORY
    )
    if justice_channel_category is not None:
        justice_channel: discord.TextChannel | None = discord.utils.get(
            justice_channel_category.text_channels, name=JUSTICE_CHANNEL_NAME
        )
        if justice_channel is not None:
            async for message in justice_channel.history(limit=1):
                if message.author == bot.user:
                    # Extract the mentions from the message
                    return [mention.id for mention in message.mentions]
    return []


@bot.event
async def on_member_join(member: discord.Member) -> None:
    """
    Standard Discord.py event handler for when a member joins the server.
    This wrapper loads the data and calls the custom implementation.

    :param member: The member who joined
    """
    async with data_lock:
        data: FullDataType = await load_data()
        if member.guild.id not in data:
            data[member.guild.id] = {}
        await _on_member_join_impl(member, data, member.guild)


async def _on_member_join_impl(
    member: discord.Member | discord.User, data: FullDataType, guild: discord.Guild
) -> None:
    """
    Event that runs when a member joins the server, welcoming them and setting their roles.
    :param guild:
    :param data:
    :param member:
    :return:
    """
    message_sent: bool = False
    welcome_dm: str = GUILDS[guild.id].welcome_dm
    if member.id not in data[guild.id]:
        message_sent = True
        data[guild.id][member.id] = MemberEntry()
        await save_data(data)
    if not hasattr(member, "guild"):
        if welcome_dm:
            logger.info(
                f"Unable to welcome user {member.display_name} (id={member.id}) because they are no longer a member. Data has been updated.",
                guild_id=guild.id,
            )
        return
    if message_sent:
        if welcome_dm:
            # DM the member
            sending_message: str = ""
            # If the member joined over 5 minutes ago
            if (discord.utils.utcnow() - member.joined_at).total_seconds() > 300:
                sending_message += "With apologies for the delay,\n"
            sending_message += welcome_dm
            if member.id != bot.user.id:
                await dm_member(member, sending_message, guild.id)
        else:
            message_sent = False
    else:
        await set_justice_role(member, await get_justice_ids(guild))

    await set_respect_role(
        guild,
        member,
        data[guild.id][member.id].shallow_score + data[guild.id][member.id].deep_score,
    )

    # Construct the appropriate message based on the situation
    welcome_status: str
    if message_sent:
        welcome_status = "welcomed to the server"
    else:
        welcome_status = "recognized as a new server member"
        if welcome_dm:
            welcome_status += (
                " (no message was sent because this isn't their first time)"
            )

    logger.info(
        f"{member.display_name} has been {welcome_status} and their roles have been set.",
        guild_id=guild.id,
    )


async def set_justice_role(member: discord.Member, justice_ids: list[int]) -> None:
    """
    Set the Justice role to the member if they are a justice.
    :param member:
    :param justice_ids:
    :return:
    """
    # If the Justice role does not exist, log the error and timestamp then return
    justice_role: discord.Role = await get_justice_role(member.guild)
    if member.id in justice_ids and not any(
        role.name == JUSTICE_ROLE_NAME for role in member.roles
    ):
        await member.add_roles(justice_role)
    elif member.id not in justice_ids and any(
        role.name == JUSTICE_ROLE_NAME for role in member.roles
    ):
        await member.remove_roles(justice_role)


def justice_score(
    server_data: ServerDataType, member: discord.Member
) -> tuple[fractions.Fraction, datetime.datetime]:
    """
    Calculate the justice score of a member. Used for sorting justices.
    :param server_data:
    :param member:
    :return:
    """
    return server_data[member.id].deep_score, member.joined_at


def is_timeout_prolongation_log(
    message: discord.Message, target_member_ids: list[int]
) -> bool:
    """
    IMPORTANT: THIS WILL NEED UPDATING IF THERE IS A CHANGE IN THE LOGGING FORMAT

    Check if the message is a timeout prolongation log.
    :param message:
    :param target_member_ids:
    :return:
    """
    if message.author.id == ELARA_LOGGER_ID:
        for embed in message.embeds:
            if (
                embed.title != "Member Timeout: Updated"
                or not any(
                    str(memb_id) in embed.fields[0].value
                    for memb_id in target_member_ids
                )
                or str(bot.user.id) not in embed.fields[1].value
                or embed.fields[3].value != ROLE_TIMEOUT_REASON
            ):
                return False
    return True


class DiscordConnectionErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out specific known network errors from discord.py logs.
        These are unavoidable for my home network and irrelevant to the bot's functionality.
        """
        if record.exc_info is None:
            return True

        exc_type = record.exc_info[0]
        error_types = (
            aiohttp.client_exceptions.ClientConnectorError,
            aiohttp.client_exceptions.WSServerHandshakeError,
            socket.gaierror,
        )

        return not (isinstance(exc_type, type) and issubclass(exc_type, error_types))


for logger_name in ("discord.client", "discord.gateway", "discord.http", "discord"):
    logging.getLogger(logger_name).addFilter(DiscordConnectionErrorFilter())


async def _smart_timeout(
    member: discord.Member,
    duration: datetime.timedelta | None,
    reason: str,
    message: str | None,
) -> bool:
    """
    Apply a smart timeout that respects other moderators' decisions

    :param member: The member to apply the timeout to.
    :param duration: The timedelta object representing the new timeout duration.
    :param reason: The reason for the timeout.
    :param message: An optional message to send to the user if the timeout is applied.
    """
    current_duration: datetime.timedelta = (
        member.timed_out_until - discord.utils.utcnow()
        if member.timed_out_until
        else datetime.timedelta()
    )

    can_update: bool = True
    if current_duration > datetime.timedelta():
        try:
            async for entry in member.guild.audit_logs(
                limit=None,
                oldest_first=False,
                action=discord.AuditLogAction.member_update,
            ):
                if entry.target.id == member.id and (
                    getattr(entry.before, "timed_out_until", None)
                    or getattr(entry.after, "timed_out_until", None)
                ):
                    can_update = (
                        entry.user.id == member.guild.me.id
                        or duration is not None
                        and duration > current_duration
                    )
                    break
        except discord.errors.Forbidden:
            logger.warning(
                "Unable to preserve moderator-issued timeouts — missing audit log permissions",
                guild_id=member.guild.id,
            )

    if can_update:
        try:
            await member.timeout(duration, reason=reason)
            if message is not None:
                await dm_member(member, message, member.guild.id)
            return True
        except discord.errors.Forbidden:
            logger.warning(
                f'Forbidden to set timeout for member "{member.display_name}" (id={member.id}).',
                guild_id=member.guild.id,
            )
    return False


async def _day_change_for_guild(guild: discord.Guild) -> None:
    """
    Perform the daily update for a single guild, refreshing scores, roles and channels.
    :param guild:
    :return:
    """
    async with data_lock:
        # Reloaded per guild: other handlers write between guilds, and a failure
        # discards only this copy.
        data: FullDataType = await load_data()

        # Create a snapshot of the member list to prevent race conditions
        members: list[discord.Member] = []

        for member in guild.members:
            if not member.bot:
                members.append(member)
                if member.id not in data[guild.id]:
                    await _on_member_join_impl(member, data, guild)
        for member_id in data[guild.id]:
            if data[guild.id][member_id].shallow_score > 0:
                data[guild.id][member_id].deep_score += fractions.Fraction(
                    math.sqrt(data[guild.id][member_id].shallow_score)
                ) / (len(members) ** (fractions.Fraction(1, 3)))
                data[guild.id][member_id].shallow_score = fractions.Fraction(0)
            elif data[guild.id][member_id].shallow_score < 0:
                data[guild.id][member_id].deep_score += data[guild.id][
                    member_id
                ].shallow_score
                data[guild.id][member_id].shallow_score /= fractions.Fraction(4, 1)
                if data[guild.id][member_id].shallow_score > fractions.Fraction(
                    -1, 100
                ):
                    data[guild.id][member_id].shallow_score = fractions.Fraction(0)
            elif data[guild.id][member_id].deep_score > fractions.Fraction(1, 10):
                data[guild.id][member_id].deep_score -= fractions.Fraction(1, 128)

            # Apply credibility decay
            data[guild.id][member_id].credibility = max(
                fractions.Fraction(0),
                data[guild.id][member_id].credibility
                - CREDIBILITY_DECAY * CREDIBILITY_RATIO,
            )

        # Calculate justices
        justices: list[discord.Member] = []
        if len(data[guild.id].keys()) >= JUSTICE_COUNT * 5:
            justices = sorted(
                members,
                key=lambda memb: justice_score(data[guild.id], memb),
                reverse=True,
            )[:JUSTICE_COUNT]
            if (
                data[guild.id][justices[-1].id].deep_score
                <= JUSTICE_DEEP_SCORE_REQUIREMENT
            ):
                justices = []

        log_deletions: list[int] = []
        for member_id in data[guild.id]:
            member = guild.get_member(member_id)

            if member is not None:
                await set_justice_role(member, [j.id for j in justices])

            # Timeout members that are missing required roles
            if member is not None and not member.bot:
                member_role_ids: set[int] = set(rl.id for rl in member.roles)
                if not all(
                    member_role_ids & role_category
                    for role_category in GUILDS[member.guild.id].required_roles
                ):
                    now = discord.utils.utcnow()
                    original_timeout_seconds: float = (
                        (member.timed_out_until - now).total_seconds()
                        if member.timed_out_until is not None
                        and member.timed_out_until > now
                        else 0.0
                    )
                    was_timed_out: bool = (
                        original_timeout_seconds
                        > TIMEOUT_NOTIFICATION_THRESHOLD.total_seconds()
                    )

                    if data[guild.id][member_id].suspended_timeout is None:
                        if await _smart_timeout(
                            member,
                            MISSING_ROLE_TIMEOUT_DURATION,
                            ROLE_TIMEOUT_REASON,
                            MISSING_ROLE_MESSAGE(was_timed_out, guild.name),
                        ):
                            data[guild.id][
                                member_id
                            ].suspended_timeout = original_timeout_seconds
                    elif await _smart_timeout(
                        member,
                        MISSING_ROLE_TIMEOUT_DURATION,
                        ROLE_TIMEOUT_REASON,
                        None,
                    ):
                        log_deletions.append(member_id)

        await save_data(data)

    # Nothing below touches the data, so it runs unlocked.

    # Make a leaderboard of the five justices
    message_content: str = ""
    i: int
    for i, justice_member in enumerate(justices):
        message_content += f"{i + 1}. {justice_member.mention}\n"
    if not message_content:
        message_content = "No justices have been determined yet."

    try:
        justice_channel_category: discord.CategoryChannel | None = discord.utils.get(
            guild.categories, name=JUSTICE_CHANNEL_CATEGORY
        )
        if justice_channel_category is None:
            justice_channel_category = await guild.create_category(
                JUSTICE_CHANNEL_CATEGORY
            )
            assert justice_channel_category is not None
        justice_channel: discord.TextChannel | None = discord.utils.get(
            justice_channel_category.text_channels, name=JUSTICE_CHANNEL_NAME
        )
        found: bool = False
        previous_justice_ids: set[int] = set()
        if justice_channel is None:
            bot_role: discord.Role | None = guild.self_role
            assert bot_role is not None, (
                f"No role is managed by the bot in guild '{guild.name}'."
            )
            justice_channel = await justice_channel_category.create_text_channel(
                JUSTICE_CHANNEL_NAME,
                overwrites={
                    guild.default_role: discord.PermissionOverwrite(
                        send_messages=False,
                        create_public_threads=False,
                        create_private_threads=False,
                    ),
                    bot_role: discord.PermissionOverwrite(send_messages=True),
                },
            )
        else:
            async for message in justice_channel.history():
                if message.author == bot.user:
                    previous_justice_ids = {
                        int(mention.id) for mention in message.mentions
                    }
                if not found and message.content != message_content:
                    await message.delete()
                else:
                    found = True
        if not found:
            # Only mention the justices that aren't in previous_justice_ids
            await justice_channel.send(
                message_content,
                allowed_mentions=discord.AllowedMentions(
                    users=[
                        discord.Object(id=justice_id)
                        for justice_id in set(justice.id for justice in justices)
                        - previous_justice_ids
                    ]
                ),
            )
    except discord.errors.Forbidden:
        logger.warning(
            f"Unable to update the justice list in `#{JUSTICE_CHANNEL_NAME}`",
            guild_id=guild.id,
        )

    # Purge the polls channel, but only if the guild has it enabled
    if GUILDS[guild.id].purge_polls:
        polls_channel: discord.TextChannel | None = discord.utils.get(
            guild.text_channels, name="polls"
        )
        if polls_channel is not None:
            try:
                deleted_count: int = len(
                    await polls_channel.purge(
                        after=datetime.datetime(year=2024, month=7, day=14),
                        check=lambda msg: msg.poll is None
                        and not msg.pinned
                        and not msg.content.startswith("[POLL]"),
                        bulk=True,
                        limit=None,
                        oldest_first=True,
                        reason="Clean up non-poll messages.",
                    )
                )
                if deleted_count > 0:
                    logger.info(
                        f"Deleted {deleted_count} non-poll messages in the polls channel.",
                        guild_id=guild.id,
                    )
            except discord.errors.Forbidden:
                logger.warning(
                    "Unable to purge polls channel `#polls`", guild_id=guild.id
                )
    logger.info("Data update complete.", guild_id=guild.id)

    if GUILDS[guild.id].logger.enabled:
        await asyncio.sleep(30)

        logger_channel: discord.TextChannel | None = discord.utils.get(
            guild.text_channels, name=GUILDS[guild.id].logger.channel_name
        )
        if logger_channel is not None:
            try:
                # Use the day_change_time of today as the after parameter
                deleted = await logger_channel.purge(
                    after=datetime.datetime.combine(
                        datetime.date.today(), DAY_CHANGE_TIME
                    ),
                    check=lambda msg: is_timeout_prolongation_log(msg, log_deletions),
                    bulk=True,
                    limit=None,
                    reason="Clean up timeout logs.",
                )
                logger.info(
                    f"Deleted {len(deleted)} role timeout prolongation logs from the logger channel.",
                    guild_id=guild.id,
                )
            except discord.errors.Forbidden:
                logger.warning(
                    f"Unable to purge logger channel `#{GUILDS[guild.id].logger.channel_name}`",
                    guild_id=guild.id,
                )
        else:
            logger.warning(
                f"Unable to access logger channel `#{GUILDS[guild.id].logger.channel_name}`",
                guild_id=guild.id,
            )


@tasks.loop(time=DAY_CHANGE_TIME)
async def day_change() -> None:
    """
    Loop that runs every day to update the data and assign roles.
    :return:
    """
    logger.info("Day change has started.")
    async with data_lock:
        backup_file_path: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../data_backup")
        )
        if not os.path.exists(backup_file_path):
            os.makedirs(backup_file_path)
        backup_file: str = (
            backup_file_path
            + f"/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        if os.path.exists(DATA_FILE):
            shutil.copyfile(DATA_FILE, backup_file)  # Backup data
        else:
            # Record the empty state so the series has no gap.
            logger.error(f"Data file {DATA_FILE} is missing.", guild_id=None)
            await save_data({}, backup_file)

    failed_guilds: list[int] = []
    for guild in guild_objects:
        assert guild is not None

        try:
            await _day_change_for_guild(guild)
        except Exception:
            failed_guilds.append(guild.id)
            logger.exception("Day change failed.", guild_id=guild.id)

    if failed_guilds:
        logger.error(
            f"Day change complete for {len(guild_objects) - len(failed_guilds)} of "
            f"{len(guild_objects)} guilds.",
            guild_id=None,
        )
    else:
        logger.info("Full day change complete.", guild_id=None)


# When a user updates their roles, check if they have the required roles
@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """
    Event that runs when a member updates their roles, checking if they have the required roles.
    :param before:
    :param after:
    :return:
    """
    if before.roles == after.roles or after.guild.id not in GUILDS:
        return
    if not all(
        set(rl.id for rl in before.roles) & role_category
        for role_category in GUILDS[before.guild.id].required_roles
    ) and all(
        set(rl.id for rl in after.roles) & role_category
        for role_category in GUILDS[after.guild.id].required_roles
    ):
        assert before.id == after.id
        async with data_lock:
            data: FullDataType = await load_data()
            if after.id not in data[after.guild.id]:
                await _on_member_join_impl(after, data, after.guild)
            if data[after.guild.id][after.id].suspended_timeout is not None:
                if data[after.guild.id][after.id].suspended_timeout > 0.0:
                    duration = datetime.timedelta(
                        seconds=data[after.guild.id][after.id].suspended_timeout
                    )
                    await _smart_timeout(
                        after,
                        duration,
                        "Resume timeout from before role-acquisition obligation.",
                        TIMEOUT_RESUME_MESSAGE(after.guild.name, duration),
                    )
                else:
                    await _smart_timeout(
                        after,
                        None,
                        "Acquired necessary roles.",
                        ROLE_RESTORATION_MESSAGE(after.guild.name),
                    )
                data[after.guild.id][after.id].suspended_timeout = None
                await save_data(data)


@bot.event
async def on_guild_join(guild: discord.Guild) -> None:
    """
    Event that runs when the bot joins a new guild, setting the bot's nickname.
    :param guild:
    """
    await update_bot_nickname(guild)
    if guild.id not in GUILDS:
        logger.info(
            f"Joined new guild—add it to {CONFIG_FILE} if you want the bot to work here.",
            guild_id=guild.id,
        )


async def shutdown() -> None:
    """
    Gracefully shuts down the bot.
    """
    logger.debug("Waiting to acquire data_lock...", guild_id=None)
    await data_lock.acquire()
    logger.debug("Data lock acquired; closing bot...", guild_id=None)
    await bot.close()  # Gracefully close the Discord bot connection
    logger.info(
        "Shutdown Complete".center(shutil.get_terminal_size().columns, "="),
        guild_id=None,
    )


# noinspection PyUnusedLocal
def signal_handler(sig: int, frame: FrameType | None) -> None:
    """
    Signal handler for SIGTERM.
    :param sig:
    :param frame:
    """
    logger.info(
        "SIGTERM Received—Shutting Down".center(
            shutil.get_terminal_size().columns, "="
        ),
        guild_id=None,
    )
    shutdown_event.set()
    loop = asyncio.get_event_loop()
    loop.create_task(shutdown())


# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)


def handle_exception(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """
    Global exception handler that restarts the script on an unhandled exception.

    :param exc_type:
    :param exc_value:
    :param exc_traceback:
    """
    # Print the error and stack trace
    logger.error("Unhandled exception occurred:", guild_id=None)
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    # If the script has been running for less than 1 hour, exit
    if time.time() - start_time < 1000:
        exit(1)

    # Optional: delay before restarting
    time.sleep(60)

    # Restart the script
    logger.info("Restarting script...", guild_id=None)
    os.execv(sys.executable, [sys.executable] + sys.argv)


# Register the global exception handler
sys.excepthook = handle_exception

# Run the bot
bot.run(TOKEN, log_handler=console_handler, log_formatter=formatter)
