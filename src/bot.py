"""
This bot is designed to manage the KaMS Club Discord server. It assigns roles to members based on their behavior and the roles they have. It also allows members to vote on other members with a severity ranging from -1 to 1. The bot will
automatically time out members with low respect scores and notify them if they have been timed out for more than a certain amount of time. The bot also assigns the Justice role to the top five members with the highest respect scores.

Due to a Discord limitation, you must restrict the /justice_toolbox command visibility to the Justice role manually. To do this, go to server settings -> integrations -> click on "KaMS Club" (manage).

Generating Discord OAuth2 Link:
- Scopes: applications.commands, bot
- Bot Permissions:
  - General: Manage Roles, Manage Channels, Ban Members, Moderate Members
  - Text: Send Messages, Send Messages in Threads, Manage Messages, Read Message History
  - Voice: None required.
"""
import asyncio
import datetime
import fractions
import json
import logging
import math
import os
import shutil
import signal
import sys
import time
import traceback
from types import FrameType
from typing import Callable

import discord
import numpy as np
from discord.ext import commands, tasks
from dotenv import load_dotenv
from scipy.interpolate import interp1d

# Set the timezone to UTC
os.environ['TZ'] = 'UTC'
time.tzset()

# Parameters
GUILD_ID: int = 1201368154174144602
JUSTICE_COUNT: int = 5
JUSTICE_CHANNEL_NAME: str = "justices"
JUSTICE_CHANNEL_CATEGORY: str = "Information"
RESPECTFUL_ROLE_NAME: str = "Respectful :)"
DISRESPECTFUL_ROLE_NAME: str = "Disrespectful :("
TIMEOUT_THRESHOLD: float = -0.3  # If a member's shallow score falls below this value, member gets timed out
TIMEOUT_NOTIFICATION_THRESHOLD: datetime.timedelta = datetime.timedelta(
    minutes=0.5)  # If a member gets timed out for more than this, member gets notified
TIMEOUT_DURATION_OUTLINE: dict[float, float] = {1.0: 0.0, 0.0: 0.0,
                                                TIMEOUT_THRESHOLD: TIMEOUT_NOTIFICATION_THRESHOLD.total_seconds() / 60.0,
                                                -1.0: 20.0, -2.0: 300.0, -3.0: 10080.0,
                                                -4.0: 10080.0}  # Score: Timeout duration (minutes)
REQUIRED_ROLES: list[set[int]] = [
    {1225900663746330795, 1225899714508226721, 1225900752225177651, 1225900807216562217, 1260753793566511174},
    {1256626845970075779, 1256627378763993189},
    {1261372426382737610, 1261371054161662044}]  # Ids of roles that are required to access the server
MISSING_ROLE_MESSAGE: Callable[[bool], str] = lambda timed_out: (
    f"Hi there. It seems like you're missing some roles, which is why {'you\'ve been temporarily timed out' if not timed_out else 'your disrespect timeout has been put on hold and will stop decreasing'}. No worries, "
    f"though! To {'regain access to the server' if not timed_out else 'keep serving your disrespect timeout until it\'s done'}, just visit the <id:customize> tab to assign yourself the necessary roles. If you have any "
    f"questions or need assistance, feel free to reach out to a moderator. We're here to help!")
ROLE_RESTORATION_MESSAGE = "You have been untimed out due to acquiring the necessary roles. Welcome back!"
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
MISSING_ROLE_TIMEOUT_DURATION: datetime.timedelta = datetime.timedelta(days=2)
JUSTICE_DEEP_SCORE_REQUIREMENT: float = 1.5
DAY_CHANGE_TIME: datetime.time = datetime.time(hour=0, minute=0, second=0)
JUSTICE_ROLE_NAME = "Justice"
ERROR_SYMBOL = ":x:"
SUCCESS_SYMBOL = ":white_check_mark:"
ELARA_LOGGER_ID: int = 1274076825009655863
LOGGER_CHANNEL_NAME: str = "logger"
ROLE_TIMEOUT_REASON: str = "Missing required roles."
CREDIBILITY_RATIO: float = 2.0e-20  # Credibility earned per second of conversation
CREDIBILITY_DECAY: int = 10  # Seconds-worth of credibility lost per day
CREDIBILITY_EARNING_EXCLUSION_CHANNELS: list[int] = [1201374063810064484, 1217615412146077806, 1263269073538515005,
                                                     1217278514298884176]


# Record start time
start_time: float = time.time()

# Load environment variables from .env file
load_dotenv()

# Load the token from an environment variable
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# ===================================================GLOBAL VARIABLES===================================================
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='', intents=intents)
data_file_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data.json"))
# Configure logging, excluding discord logs
logger = logging.getLogger('kams-bot')
logger.setLevel(logging.INFO)
# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
formatter = logging.Formatter(LOGGING_FORMAT)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
guild_object: discord.Guild | None = None
is_initialized = False
# Extract x and y coordinates from the dictionary
x_coords: np.ndarray = np.array(list(TIMEOUT_DURATION_OUTLINE.keys()))
y_coords: np.ndarray = np.array(list(TIMEOUT_DURATION_OUTLINE.values()))
# Create a linear interpolation function
linear_interp = interp1d(x_coords, y_coords, fill_value='extrapolate')  # linear interpolation
# Generate points to plot the function
x_values: np.ndarray = np.linspace(min(x_coords), max(x_coords), 500)
y_values: np.ndarray = linear_interp(x_values)


# ===================================================UTILITY FUNCTIONS==================================================
# Function to evaluate the linear interpolation at any given x
def calculate_timeout(x: fractions.Fraction) -> float:
    """
    Calculate the timeout duration based on the shallow score.
    :param x:
    :return:
    """
    return float(linear_interp(float(x)))


class MemberEntry:
    """
    Class to represent a member entry in the data file.
    """

    def __init__(self, shallow_score: fractions.Fraction = fractions.Fraction(0),
                 deep_score: fractions.Fraction = fractions.Fraction(0),
                 credibility: fractions.Fraction = fractions.Fraction(0),
                 opinions: dict[int, fractions.Fraction] | None = None,
                 latest_message_time: float = discord.utils.DISCORD_EPOCH / 1000,
                 conversation_start_time: float = discord.utils.DISCORD_EPOCH / 1000,
                 suspended_timeout: float | None = None) -> None:
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
            opinions={int(k): fractions.Fraction(v) for k, v in data.get("opinions", {}).items()},
            latest_message_time=data.get("latest_message_time", discord.utils.DISCORD_EPOCH / 1000),
            conversation_start_time=data.get("conversation_start_time", discord.utils.DISCORD_EPOCH / 1000),
            suspended_timeout=data.get("suspended_timeout")
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
            "suspended_timeout": self.suspended_timeout
        }


# Define the type for the data structure
DataType = dict[int, MemberEntry]

# Initialize a lock for thread-safe file access
data_lock = asyncio.Lock()


async def load_data() -> DataType:
    """
    Load data from the JSON file asynchronously.
    :return: The data dictionary.
    """
    async with data_lock:
        if os.path.exists(data_file_path):
            try:
                with open(data_file_path, "r", encoding="utf-8") as file:
                    return {int(key): MemberEntry.from_dict(value) for key, value in json.load(file).items()}
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading data: {e}")
        return {}


async def save_data(data: DataType, output_file: str = data_file_path) -> None:
    """
    Save data to the JSON file asynchronously.
    :param data: The data dictionary to save.
    :param output_file: Path to the output JSON file.
    """
    async with data_lock:
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump({str(key): value.to_dict() for key, value in data.items()}, file, indent=2)
        except IOError as e:
            print(f"Error saving data: {e}")


async def set_respect_role(guild: discord.Guild, member: discord.Member, score: fractions.Fraction) -> None:
    """
    Set the respect role based on the score.
    :param guild:
    :param member:
    :param score:
    :return:
    """
    disrespectful_role: discord.Role | None = discord.utils.get(guild.roles, name=DISRESPECTFUL_ROLE_NAME)
    respectful_role: discord.Role | None = discord.utils.get(guild.roles, name=RESPECTFUL_ROLE_NAME)

    if disrespectful_role is None or respectful_role is None:
        logger.error(f"The '{DISRESPECTFUL_ROLE_NAME}' or '{RESPECTFUL_ROLE_NAME}' role does not exist.")
        return

    if score >= 0.0:
        if disrespectful_role in member.roles:
            await member.remove_roles(disrespectful_role, reason=f"Respect score of {score} is positive.")
        if respectful_role not in member.roles:
            await member.add_roles(respectful_role, reason=f"Respect score of {score} is positive.")
            logger.info(f"{member.display_name} has been upgraded to '{RESPECTFUL_ROLE_NAME}'.")
    elif disrespectful_role not in member.roles and respectful_role not in member.roles:
        await member.add_roles(disrespectful_role, reason=f"Bad respect score.")
        logger.info(
            f"{member.display_name} has been assigned '{DISRESPECTFUL_ROLE_NAME}' because their roles were missing and their respect score is negative.")
    elif score < min(-1.0, -0.01 * sum(not memb.bot for memb in guild.members)):
        if respectful_role in member.roles:
            await member.remove_roles(respectful_role, reason=f"Respect score of {score} is unacceptably bad.")
            if disrespectful_role not in member.roles:
                await member.add_roles(disrespectful_role)
                logger.info(f"{member.display_name} has been downgraded to '{DISRESPECTFUL_ROLE_NAME}'.")


@bot.event
async def on_message(message: discord.Message) -> None:
    """

    :param message:
    """
    if not is_initialized:
        return
    # update the user's [latest_message_time] and [conversation_start_time] in the data file
    author_id: int = message.author.id
    if message.author == bot.user and message.mentions:
        author_id = message.mentions[0].id

    async with data_lock:
        data: DataType = await load_data()
        if author_id not in data:
            await on_member_join(message.author, data)

        # Get the timestamp of the message (use edited_at if available, else use created_at)
        message_timestamp: float = message.edited_at.timestamp() if message.edited_at else message.created_at.timestamp()

        # Check if the difference in time is greater than 300 seconds (5 minutes)
        if message_timestamp - data[author_id].latest_message_time > 300:
            # Update credibility based on the time difference and reset conversation start time
            data[author_id].credibility += (message_timestamp - data[
                author_id].conversation_start_time) / CREDIBILITY_RATIO
            data[author_id].conversation_start_time = message_timestamp

        data[author_id].latest_message_time = message_timestamp
        await save_data(data)


class JusticeToolboxView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Set Slowmode", style=discord.ButtonStyle.primary)
    async def set_slowmode(self, interaction: discord.Interaction, _button: discord.ui.Button):
        """
        :param interaction:
        :param _button:
        """
        modal = SetSlowmodeModal()
        # noinspection PyUnresolvedReferences
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Request Ban/Unban", style=discord.ButtonStyle.danger)
    async def request_ban(self, interaction: discord.Interaction, _button: discord.ui.Button):
        """

        :param interaction:
        :param _button:
        """
        select = BanUnbanView()
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message("Select whether you would like request to ban or to unban a user.",
                                                view=select, ephemeral=True)


class SetSlowmodeModal(discord.ui.Modal, title="Set Slowmode for the Current Channel"):
    length = discord.ui.TextInput(label="Length, seconds", required=True)
    reset_time = discord.ui.TextInput(label="Reset Time, minutes (optional)", required=False)

    async def on_submit(self, interaction: discord.Interaction):
        # Validate input for length
        try:
            length = int(self.length.value)
            if length < 0:
                raise ValueError("Slowmode must be a non-negative integer.")
        except ValueError:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                content=f"{ERROR_SYMBOL} Invalid slowmode length. Slowmode must be a non-negative integer.")
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
                    content=f"{ERROR_SYMBOL} Invalid reset time. Please enter a positive number.")
                return

        if length < 0:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(content=f"{ERROR_SYMBOL} Slowmode must be a non-negative integer.")
            return
        try:
            await interaction.channel.edit(slowmode_delay=length,
                                           reason=f"Set by user {interaction.user.id} (\"{interaction.user.display_name}\") via Justice Toolbox")
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                content=f"{SUCCESS_SYMBOL} Slowmode for {interaction.channel.mention} has been set to {length} second{"s" if length != 1 else ""} {f"with a reset time of {reset_time} minutes" if reset_time is not None else ''}.")
            if reset_time is not None:
                await asyncio.sleep(reset_time * 60.0)
                await interaction.channel.edit(slowmode_delay=0,
                                               reason=f"Reset from command by user {interaction.user.id} via Justice Toolbox")
        except discord.errors.Forbidden:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                f"{ERROR_SYMBOL} I do not have permission to set slowmode in this channel.")


BanRequestsType = dict[int, dict[int, dict[str, bool | str]]]  # user_id: {requester_id: ban_request}


class RequestBanModal(discord.ui.Modal):

    def __init__(self, ban_bool: bool, **kwargs):
        title = "Request Ban" if ban_bool else "Request Unban"
        super().__init__(title=title, **kwargs)
        self.ban_bool = ban_bool
        self.target: discord.ui.TextInput = discord.ui.TextInput(
            label=f"User ID to {"Ban" if self.ban_bool else "Unban"}", required=True, placeholder="012345678910111213",
            style=discord.TextStyle.short)
        self.reason = discord.ui.TextInput(label=f"Reason for {"Ban" if self.ban_bool else "Unban"}",
                                           style=discord.TextStyle.paragraph, required=True, min_length=60)

        # Add the text inputs to the modal
        self.add_item(self.target)
        self.add_item(self.reason)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            target_object: discord.User = await bot.fetch_user(int(self.target.value))
        except (ValueError, discord.errors.NotFound, discord.errors.HTTPException):
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(
                content=f"{ERROR_SYMBOL} Invalid user ID. Please enter a valid user ID.")
            return

        reason = self.reason.value
        # Open file ban_requests.json, create it if it doesn't exist
        ban_requests: BanRequestsType = read_ban_requests()
        if target_object.id not in ban_requests:
            ban_requests[target_object.id] = {}
        existing_request: dict[str, bool | str] | None = ban_requests[target_object.id].get(interaction.user.id)
        request_changed: bool = existing_request['request'] != self.ban_bool if existing_request else False
        ban_requests[target_object.id][interaction.user.id] = {"request": self.ban_bool, "reason": reason}
        save_ban_requests(ban_requests)
        if self.ban_bool:
            # If all current justices have requested a ban, ban the user
            justice_ids: list[int] = await get_justice_ids(interaction.guild)
            if len(justice_ids) == JUSTICE_COUNT and all(
                    justice_id in ban_requests[target_object.id] and ban_requests[target_object.id][justice_id] for
                    justice_id in justice_ids):
                # Ban the user
                try:
                    await interaction.guild.ban(target_object, reason="Requested by justices.")
                except discord.errors.Forbidden:
                    # noinspection PyUnresolvedReferences
                    await interaction.response.edit_message(f"{ERROR_SYMBOL} Sorry, I am unable to ban this user.")
        else:
            # If 2/3 of current justices have requested an unban, unban the user
            justice_ids: list[int] = await get_justice_ids(interaction.guild)
            if sum(1 for justice_id in justice_ids if
                   justice_id in ban_requests[target_object.id] and not ban_requests[target_object.id][
                       justice_id]) >= 2 * len(justice_ids) / 3 and any(
                ban.user.id == target_object.id for ban in [ban async for ban in interaction.guild.bans()]):
                # Unban the user
                await interaction.guild.unban(target_object, reason="Requested by justices.")
        # noinspection PyUnresolvedReferences
        await interaction.response.edit_message(
            content=f"{SUCCESS_SYMBOL} {f"{'Unb' if not self.ban_bool else 'B'}an r" if not request_changed else "R"}equest {'submitted' if not existing_request else 'updated' + (f' from **{"ban" if existing_request["request"] else "unban"}** to **{"ban" if self.ban_bool else "unban"}**' if request_changed else '')} for user {self.target.value} (\"{target_object.display_name}\").")


class BanUnbanSelect(discord.ui.Select):
    def __init__(self):
        # To make it a multi-select dropdown, add the parameter max_values=2
        options = [discord.SelectOption(label="Ban", value="ban"), discord.SelectOption(label="Unban", value="unban"),
                   discord.SelectOption(label="Cancel Prior Request", value="cancel")]
        super().__init__(placeholder="Select an action", options=options)

    async def callback(self, interaction: discord.Interaction):
        # Save the user's choice in the interaction or context
        choice = self.values[0]
        if choice == "cancel":
            ban_requests: BanRequestsType = read_ban_requests()
            user_requests: list[tuple[int, str, bool]] = []
            for target_id in ban_requests:
                if interaction.user.id in ban_requests[target_id]:
                    target_object: discord.User = await bot.fetch_user(target_id)
                    user_requests.append((target_id, target_object.display_name,
                                          ban_requests[target_id][interaction.user.id]["request"]))
            if not user_requests:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(f"{ERROR_SYMBOL} You have no ban requests to cancel.",
                                                        ephemeral=True, delete_after=15)
            else:
                select = BanRequestCancelSelect(requests=user_requests)
                view: discord.ui.View = discord.ui.View()
                view.add_item(select)
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message("Select a ban request to cancel.", view=view, ephemeral=True,
                                                        delete_after=60)
            return
        # Pass the choice to the modal
        modal = RequestBanModal(ban_bool=choice == "ban")
        # noinspection PyUnresolvedReferences
        await interaction.response.send_modal(modal)


class BanRequestCancelSelect(discord.ui.Select):
    def __init__(self, requests: list[tuple[int, str, bool]]):
        # Ban/Unban [User id] ("display name")
        options = [discord.SelectOption(label=f'{"Ban" if request[2] else "Unban"} {request[0]} ("{request[1]}")',
                                        value=str(request[0])) for request in requests]
        super().__init__(placeholder="Select a ban request to cancel", options=options)
        self.requests = requests

    async def callback(self, interaction: discord.Interaction):
        # Delete the ban request from the file
        target_id = int(self.values[0])
        ban_requests: BanRequestsType = read_ban_requests()
        for requester_id in ban_requests[target_id]:
            if requester_id == interaction.user.id:
                ban_requests[target_id].pop(requester_id)
                save_ban_requests(ban_requests)
                # noinspection PyUnresolvedReferences
                # Find the user id in the self.requests list and get the display name
                await interaction.response.send_message(
                    f"{SUCCESS_SYMBOL} Ban request for user {target_id} (\"{next(request[1] for request in self.requests if request[0] == target_id)}\") has been cancelled.",
                    ephemeral=True)
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
            return {int(key): {int(subkey): value for subkey, value in subdict.items()} for key, subdict in
                    json.load(file).items()}
    return {}


def save_ban_requests(ban_requests: BanRequestsType) -> None:
    """
    Save the ban requests to the JSON file.
    :param ban_requests:
    """
    with open("../ban_requests.json", "w") as file:
        json.dump({str(key): {str(subkey): value for subkey, value in subdict.items()} for key, subdict in
                   ban_requests.items()}, file, indent=2)


async def dm_member(member: discord.Member, message: str) -> None:
    """
    Send a direct message to a member, creating a DM channel if necessary.
    :param member:
    :param message:
    """

    try:
        await member.send(message)
    except discord.errors.Forbidden:
        logger.error(f"Forbidden to send message to \"{member.display_name}\" (id={member.id}).")


async def get_messages_from_channel(
        channel: discord.TextChannel | discord.VoiceChannel | discord.ForumChannel | discord.CategoryChannel,
        after: datetime.datetime, output: list[discord.Message]):
    """
    Helper function to process messages from different channel types.

    :param channel: The channel to process (could be TextChannel, VoiceChannel, or ForumChannel).
    :param after: The timestamp to start retrieving messages from.
    :param output: The list to store messages.
    """
    match channel:
        case discord.TextChannel():
            # Process TextChannel messages
            async for message in channel.history(after=after, limit=None):
                output.append(message)

            for thread in channel.threads:
                async for message in thread.history(after=after, limit=None):
                    output.append(message)

        case discord.VoiceChannel():
            # Process VoiceChannel messages (no threads)
            async for message in channel.history(after=after, limit=None):
                output.append(message)

        case discord.ForumChannel():
            # Process ForumChannel threads
            for thread in channel.threads:
                async for message in thread.history(after=after, limit=None):
                    output.append(message)

        case discord.CategoryChannel():
            # Process subchannels within CategoryChannel
            for subchannel in channel.channels:
                await get_messages_from_channel(subchannel, after, output)


@bot.event
async def on_ready() -> None:
    """
    Event that runs when the bot is ready, syncing the commands and starting the day_change loop.
    """

    global is_initialized
    if is_initialized:
        return

    global guild_object
    # Get the guild object
    guild_object = bot.get_guild(GUILD_ID)
    if guild_object is None:
        logger.error("Could not find provided guild.")
        exit(1)

    @bot.tree.command(name="justice_toolbox", description="Access the Justice Toolbox.", guild=guild_object)
    async def slash_justice_toolbox(interaction: discord.Interaction) -> None:
        """
        Access the Justice Toolbox.
        :param interaction:
        """
        justice_role: discord.Role | None = discord.utils.get(interaction.guild.roles, name=JUSTICE_ROLE_NAME)
        if justice_role is None:
            logger.error("The 'Justice' role does not exist in the guild. Error accessing it for the Justice Toolbox.")
            return
        if justice_role not in interaction.user.roles:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message("You must be a Justice to access the Justice Toolbox.",
                                                    ephemeral=True)
            return
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message("Justice Toolbox", view=JusticeToolboxView(), ephemeral=True)

    @bot.tree.command(name="my_opinions", description="View your opinions, constructed from your votes.",
                      guild=guild_object)
    async def slash_my_opinions(interaction: discord.Interaction) -> None:
        """
        Output a table of percentages, adding to <= 1
        """
        output = ""
        async with data_lock:
            data: DataType = await load_data()
            if interaction.user.id not in data:
                await on_member_join(interaction.user)

            if len(data[interaction.user.id].opinions) == 0:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message("You have not voted on anyone yet.", ephemeral=True)
                return
            for target_id, severity in data[interaction.user.id].opinions.items():
                target: discord.User = await bot.fetch_user(target_id)
                output += f"**{target.display_name}**: {severity}\n"
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message(output, ephemeral=True)

    @bot.tree.command(name="vote",
                      description="Vote for a user with a severity ranging from -1 to 1. See The Rules for more information.",
                      guild=guild_object)
    async def slash_vote(interaction: discord.Interaction, target: discord.User, severity: float, reason: str,
                         hidden: bool) -> None:
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
        print(fraction_severity)
        if fraction_severity == 0.0:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message("You cannot vote with a severity of 0.", ephemeral=True)
            return
        async with data_lock:
            data: DataType = await load_data()
            if interaction.user.id not in data:
                await on_member_join(interaction.user)
            target_member: discord.Member | None = interaction.guild.get_member(target.id)
            if target.id not in data and target_member is not None:
                await on_member_join(target_member)
            if -1.0 > fraction_severity or fraction_severity > 1.0:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message("Invalid severity value. Please use a value between -1 and 1.",
                                                        ephemeral=True)
                logger.info(
                    f"Invalid severity value for {interaction.user.display_name} to vote for {target.display_name} with severity {fraction_severity}.")
                return
            data[interaction.user.id].opinions[target.id] = (
                    data[interaction.user.id].opinions[target.id] + fraction_severity) if target.id in data[
                interaction.user.id].opinions else fraction_severity

            # And adjust the rest of the user's opinions to make sure their absolute sum is less than or equal to 1
            adjust_factor: fractions.Fraction = fractions.Fraction(1, max(1, sum(
                abs(value) for value in data[interaction.user.id].opinions.values())))
            fraction_severity *= adjust_factor
            for key in data[interaction.user.id].opinions:
                data[interaction.user.id].opinions[key] *= adjust_factor
            assert sum(map(abs, data[interaction.user.id].opinions.values())) <= 1.0

            data[target.id].shallow_score = data[target.id].shallow_score + fraction_severity * max(
                data[interaction.user.id].credibility, fractions.Fraction(1, 100))
            if target_member is not None:
                await set_respect_role(interaction.guild, target_member,
                                       data[target.id].shallow_score + data[target.id].deep_score)
                if data[target.id].shallow_score < (TIMEOUT_THRESHOLD + 1.0):
                    # Timeout procedure
                    timeout_minutes = calculate_timeout(
                        data[target.id].shallow_score + min(data[target.id].deep_score, fractions.Fraction(1, 2)))
                    old_duration: datetime.timedelta = datetime.timedelta()
                    if target_member.timed_out_until is not None and (
                            target_member.timed_out_until - discord.utils.utcnow()) > old_duration:
                        old_duration = target_member.timed_out_until - discord.utils.utcnow()
                    new_duration: datetime.timedelta = datetime.timedelta(minutes=timeout_minutes)
                    if (fraction_severity < 0 or new_duration < old_duration) and new_duration != old_duration:
                        until: datetime.datetime = discord.utils.utcnow() + new_duration
                        if data[target_member.id].suspended_timeout is not None:
                            data[target_member.id].suspended_timeout = new_duration.total_seconds()
                        else:
                            try:
                                await target_member.edit(timed_out_until=until,
                                                         reason=f"Voted {fraction_severity} by a member.")
                                logger.info(
                                    f"{target_member.display_name} has been timed out for {timeout_minutes} minutes.")
                                if old_duration < TIMEOUT_NOTIFICATION_THRESHOLD < new_duration:
                                    await dm_member(target_member,
                                                    f"You have been timed out for {timeout_minutes} minutes due to your low respect score. Please take this time to reflect on your behavior. If you have any questions, feel free to reach out "
                                                    f"to a "
                                                    f"moderator.")
                            except discord.errors.Forbidden:
                                logger.error(
                                    f"Forbidden to timeout user \"{target_member.display_name}\" (id={target_member.id}).")

            await save_data(data)

            if not hidden:
                # Send a message publicly
                public_message: str = f"{interaction.user.mention} has {'up' if fraction_severity > 0 else 'down'}voted {target.mention} with severity {fraction_severity}. Reason: {reason}"  # If changing this line, also update on_message.
                await interaction.channel.send(public_message)
            try:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(
                    f"Vote successful! Your opinion on {target.display_name} is now {data[interaction.user.id].opinions[target.id]}",
                    ephemeral=True, delete_after=15)
            except discord.errors.NotFound:
                logger.error(
                    f"Interaction not found to send vote confirmation to \"{interaction.user.display_name}\". Processing may have taken too long. Proceeding to send a DM.")
                await dm_member(interaction.user,
                                f"With apologies for the delay, your vote for {target.display_name} with severity {fraction_severity} has been successfully processed. Your opinion on {target.display_name} is now "
                                f"{data[interaction.user.id].opinions[target.id]}.")

    async with (data_lock):
        logger.info("Bot is ready, starting to sync commands...")
        commands_synced: list[discord.app_commands.AppCommand] = await bot.tree.sync(guild=guild_object)
        assert not bot.tree.get_commands()
        assert len(commands_synced) == len(bot.tree.get_commands(guild=guild_object))
        logger.info("Slash commands synced!")
        day_change.start()
        logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
        logger.info("Catching up on missed messages...")
        # Load the data
        data_records: DataType = await load_data()
        # Get all the missed messages
        missed_messages: list[discord.Message] = []
        after_time: datetime.datetime = datetime.datetime.fromtimestamp(
            max(entry.latest_message_time for entry in
                data_records.values()) if data_records else discord.utils.DISCORD_EPOCH / 1000)
        for channel in guild_object.channels:
            if channel not in CREDIBILITY_EARNING_EXCLUSION_CHANNELS:
                channel_messages = []
                await get_messages_from_channel(channel, after_time, channel_messages)
                missed_messages.extend(channel_messages)
    # Sort
    missed_messages.sort(key=lambda msg: msg.created_at)
    for message in missed_messages:
        # Process the message
        await on_message(message)
    is_initialized = True
    logger.info("Initialization complete.")


async def get_justice_ids(guild: discord.Guild) -> list[int]:
    """
    Get the justice ids from the justices channel.

    :return:
    """
    # Fetch justice ids from the justices channel
    justice_channel_category: discord.CategoryChannel | None = discord.utils.get(guild.categories,
                                                                                 name=JUSTICE_CHANNEL_CATEGORY)
    if justice_channel_category is not None:
        justice_channel: discord.TextChannel | None = discord.utils.get(justice_channel_category.text_channels,
                                                                        name=JUSTICE_CHANNEL_NAME)
        if justice_channel is not None:
            async for message in justice_channel.history(limit=1):
                if message.author == bot.user:
                    # Extract the mentions from the message
                    return [mention.id for mention in message.mentions]
    return []


@bot.event
async def on_member_join(member: discord.Member, data: DataType | None = None) -> None:
    """
    Event that runs when a member joins the server, welcoming them and setting their roles.
    :param data:
    :param member:
    :return:
    """
    # Ensure the member is not a bot
    if member.bot:
        return
    message_sent: bool = False
    locked: bool = False
    if data is None:
        data = await load_data()
        await data_lock.acquire()
        locked = True
    if not member.id in data:
        # DM the member
        sending_message: str = ""
        # If the member joined over 5 minutes ago
        if (discord.utils.utcnow() - member.joined_at).total_seconds() > 300:
            sending_message += "With apologies for the delay,\n"
        sending_message += (
            "Welcome to the KaMS Club Discord server! As you may have noticed in the rules, your nickname must include your real-life name. Please make sure to update your nickname accordingly if you haven't already. "
            "Thanks!")
        await dm_member(member, sending_message)
        message_sent = True

        data[member.id] = MemberEntry()
        await save_data(data)
    else:
        await set_justice_role(member, await get_justice_ids(member.guild))

    await set_respect_role(member.guild, member, data[member.id].shallow_score + data[member.id].deep_score)
    if locked:
        data_lock.release()
    logger.info(
        f"{member.display_name} has been welcomed to the server {"(no message was sent because this isn't their first time) " if not message_sent else ""}and their roles have been set.")


async def set_justice_role(member: discord.Member, justice_ids: list[int]) -> None:
    """
    Set the Justice role to the member if they are a justice.
    :param member:
    :param justice_ids:
    :return:
    """
    # If the Justice role does not exist, log the error and timestamp then return
    justice_role: discord.Role | None = discord.utils.get(member.guild.roles, name=JUSTICE_ROLE_NAME)
    if justice_role is None:
        logger.error("The 'Justice' role does not exist in guild \"{member.guild.name}\".")
        return
    if member.id in justice_ids and not any(role.name == JUSTICE_ROLE_NAME for role in member.roles):
        await member.add_roles(justice_role)
    elif member.id not in justice_ids and any(role.name == JUSTICE_ROLE_NAME for role in member.roles):
        await member.remove_roles(justice_role)


def justice_score(data: DataType, member: discord.Member) -> tuple[fractions.Fraction, datetime.datetime]:
    """
    Calculate the justice score of a member. Used for sorting justices.
    :param data:
    :param member:
    :return:
    """
    return data[member.id].deep_score, member.joined_at


def is_timeout_prolongation_log(message: discord.Message, target_member_ids: list[
    int]) -> bool:  # THIS WILL NEED UPDATING IF THERE IS A CHANGE IN THE LOGGING FORMAT
    """
    Check if the message is a timeout prolongation log.
    :param message:
    :param target_member_ids:
    :return:
    """
    if message.author.id == ELARA_LOGGER_ID:
        for embed in message.embeds:
            if embed.title != "Member Timeout: Updated" or not any(
                    str(memb_id) in embed.fields[0].value for memb_id in target_member_ids) or str(bot.user.id) not in \
                    embed.fields[1].value or embed.fields[3].value != ROLE_TIMEOUT_REASON:
                return False
    return True


@tasks.loop(time=DAY_CHANGE_TIME)
async def day_change() -> None:
    """
    Loop that runs every day at midnight to update the data and assign roles.
    :return:
    """
    logger.info("Day change has started.")
    async with data_lock:
        data: DataType = await load_data()
        backup_file_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_backup"))
        if not os.path.exists(backup_file_path):
            os.makedirs(backup_file_path)
        await save_data(data,
                        backup_file_path + f"/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")  # Backup data

        guild: discord.Guild | None = bot.get_guild(GUILD_ID)
        if guild is None:
            logger.error("Could not find provided guild.")
            return
        member_count: int = 0
        for member in guild.members:
            if not member.bot:
                member_count += 1
                if member.id not in data:
                    await on_member_join(member, data)
        for member_id in data:
            if data[member_id].shallow_score > 0:
                data[member_id].deep_score += math.sqrt(data[member_id].shallow_score) / (member_count ** (1 / 3))
                data[member_id].shallow_score = 0.0
            elif data[member_id].shallow_score < 0:
                data[member_id].deep_score += data[member_id].shallow_score
                data[member_id].shallow_score /= 4.0
                if data[member_id].shallow_score > -0.01:
                    data[member_id].shallow_score = 0.0
            elif data[member_id].deep_score > 0.1:
                data[member_id].deep_score -= 0.0078125  # 1/28

            # Apply credibility decay
            data[member_id].credibility = max(fractions.Fraction(0), data[member_id].credibility - CREDIBILITY_DECAY)

        # Calculate justices
        justices: list[discord.Member] = []
        if len(data.keys()) >= JUSTICE_COUNT * 5:
            justices = sorted(guild.members, key=lambda memb: justice_score(data, memb), reverse=True)[:JUSTICE_COUNT]
            if data[justices[-1].id].deep_score <= JUSTICE_DEEP_SCORE_REQUIREMENT:
                justices = []

        log_deletions: list[int] = []
        for member_id in data:
            member = guild.get_member(member_id)

            if member is not None:
                await set_justice_role(member, [j.id for j in justices])

            # Timeout members that are missing required roles
            if member is not None and not member.bot:
                member_role_ids: set[int] = set(rl.id for rl in member.roles)
                if not all(member_role_ids & role_category for role_category in REQUIRED_ROLES):
                    try:
                        was_timed_out: bool = member.timed_out_until is not None and member.timed_out_until > discord.utils.utcnow()
                        await member.timeout(MISSING_ROLE_TIMEOUT_DURATION, reason="Missing required roles.")
                        if data[member_id].suspended_timeout is not None:
                            data[member_id].suspended_timeout = 0.0 if not was_timed_out else max(0.0, (
                                    member.timed_out_until - discord.utils.utcnow()).total_seconds())
                            await dm_member(member, MISSING_ROLE_MESSAGE(was_timed_out))
                            logger.info(
                                f"{member.display_name} (id={member_id}) has been timed out for {MISSING_ROLE_TIMEOUT_DURATION.total_seconds() / 86400.0} days due to missing required roles.")
                        else:
                            log_deletions.append(member_id)
                    except discord.errors.Forbidden:
                        logger.error(
                            f"Forbidden to timeout user \"{member.display_name}\" (id={member_id}) for missing required roles.")

        await save_data(data)
    logger.info("Data update complete.")

    # Make a leaderboard of the five justices
    message_content: str = ""
    i: int
    for i, justice_member in enumerate(justices):
        message_content += f"{i + 1}. {justice_member.mention}\n"
    if not message_content:
        message_content = "No justices have been determined yet."

    justice_channel_category: discord.CategoryChannel | None = discord.utils.get(guild.categories,
                                                                                 name=JUSTICE_CHANNEL_CATEGORY)
    if justice_channel_category is None:
        justice_channel_category = await guild.create_category(JUSTICE_CHANNEL_CATEGORY)
        assert justice_channel_category is not None
    justice_channel: discord.TextChannel | None = discord.utils.get(justice_channel_category.text_channels,
                                                                    name=JUSTICE_CHANNEL_NAME)
    found: bool = False
    previous_justice_ids: set[int] = set()
    if justice_channel is None:
        justice_channel = await justice_channel_category.create_text_channel(JUSTICE_CHANNEL_NAME, overwrites={
            guild.default_role: discord.PermissionOverwrite(send_messages=False, create_public_threads=False,
                                                            create_private_threads=False),
            discord.utils.get(guild.roles, name="KaMS Club"): discord.PermissionOverwrite(send_messages=True)})
    else:
        async for message in justice_channel.history():
            if message.author == bot.user:
                previous_justice_ids = {int(mention.id) for mention in message.mentions}
            if not found and message.content != message_content:
                await message.delete()
            else:
                found = True
    if not found:
        # Only mention the justices that aren't in previous_justice_ids
        await justice_channel.send(message_content, allowed_mentions=discord.AllowedMentions(
            users=[discord.Object(id=justice_id) for justice_id in
                   set(justice.id for justice in justices) - previous_justice_ids]))

    polls_channel: discord.TextChannel | None = discord.utils.get(guild.text_channels, name="polls")
    if polls_channel is not None:
        deleted_count: int = len(
            await polls_channel.purge(after=datetime.datetime(year=2024, month=7, day=14), check=lambda
                msg: msg.poll is None and not msg.pinned and not msg.content.startswith("[POLL]"), bulk=True,
                                      limit=None, oldest_first=True,
                                      reason="Clean up non-poll messages."))
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} non-poll messages in the polls channel.")
    await asyncio.sleep(30)

    # Purge the log_deletions from the logger channel
    logger_channel: discord.TextChannel | None = discord.utils.get(guild.text_channels, name=LOGGER_CHANNEL_NAME)
    if logger_channel is not None:
        # Use the day_change_time of today as the after parameter
        deleted = await logger_channel.purge(after=datetime.datetime.combine(datetime.date.today(), DAY_CHANGE_TIME),
                                             check=lambda msg: is_timeout_prolongation_log(msg, log_deletions),
                                             bulk=True, limit=None, reason="Clean up timeout logs.")
        logger.info(f"Deleted {len(deleted)} role timeout prolongation logs from the logger channel.")
    logger.info("Full day change complete.")


# When a user updates their roles, check if they have the required roles
@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """
    Event that runs when a member updates their roles, checking if they have the required roles.
    :param before:
    :param after:
    :return:
    """
    if before.roles == after.roles:
        return
    if not all(set(rl.id for rl in before.roles) & role_category for role_category in REQUIRED_ROLES) and all(
            set(rl.id for rl in after.roles) & role_category for role_category in REQUIRED_ROLES):
        assert before.id == after.id
        async with data_lock:
            data: DataType = await load_data()
            if not after.id in data:
                await on_member_join(after)
            if data[after.id].suspended_timeout is not None:
                try:
                    if data[after.id].suspended_timeout > 0.0:
                        await after.timeout(datetime.timedelta(seconds=data[after.id].suspended_timeout),
                                            reason="Resume timeout from before role-acquisition obligation.")
                    else:
                        await after.timeout(None, reason="Acquired necessary roles.")
                except discord.errors.Forbidden:
                    logger.error(
                        f"Forbidden to untimeout user \"{after.display_name}\" (id={after.id}) for role acquisition.")
                    return
                logger.info(
                    f"{after.display_name} (id={after.id}) has been untimed out due to acquiring the necessary roles.")
                if after.timed_out_until is not None and (
                        after.timed_out_until - discord.utils.utcnow()) > TIMEOUT_NOTIFICATION_THRESHOLD:
                    logger.info(
                        f"{after.display_name} (id={after.id}) still has a respect timeout of {data[after.id].suspended_timeout / 60.0} minutes to serve.")
                    await dm_member(after,
                                    f"Your role timeout has been removed, but you still have a timeout of {data[after.id].suspended_timeout / 60.0} minutes to serve.")
                else:
                    await dm_member(after, ROLE_RESTORATION_MESSAGE)
                data[after.id].suspended_timeout = None
                await save_data(data)


async def shutdown() -> None:
    """
    Gracefully shuts down the bot.
    """
    logger.debug("Waiting to acquire data_lock...")
    await data_lock.acquire()
    logger.debug("Data lock acquired; closing bot...")
    await bot.close()  # Gracefully close the Discord bot connection
    logger.info('Shutdown Complete'.center(shutil.get_terminal_size().columns, '='))


# noinspection PyUnusedLocal
def signal_handler(sig: int, frame: FrameType | None) -> None:
    """
    Signal handler for SIGTERM.
    :param sig:
    :param frame:
    """
    logger.info('SIGTERM Received—Shutting Down'.center(shutil.get_terminal_size().columns, '='))
    loop = asyncio.get_event_loop()
    loop.create_task(shutdown())


# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler that restarts the script on an unhandled exception.

    :param exc_type:
    :param exc_value:
    :param exc_traceback:
    """
    # Print the error and stack trace
    print("Unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    # If the script has been running for less than 1 hour, exit
    if time.time() - start_time < 1000:
        exit(1)

    # Optional: delay before restarting
    time.sleep(60)

    # Restart the script
    print("Restarting script...")
    os.execv(sys.executable, ['python3.12', __file__])


# Register the global exception handler
sys.excepthook = handle_exception

# Run the bot
bot.run(TOKEN, log_handler=console_handler, log_formatter=formatter)
