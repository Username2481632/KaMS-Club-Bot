"""
This bot is designed to manage the KaMS Club Discord server. It assigns roles to members based on their behavior and the roles they have. It also allows members to vote on other members with a severity ranging from -1 to 1. The bot will
automatically time out members with low respect scores and notify them if they have been timed out for more than a certain amount of time. The bot also assigns the Justice role to the top five members with the highest respect scores.

Due to a Discord limitation, you must restrict the /justice_toolbox command visibility to the Justice role manually. To do this, go to server settings -> integrations -> click on "KaMS Club" (manage).
"""
import asyncio
import datetime
import json
import logging
import math
import os
import shutil
import signal
import sys
import time
import traceback
from decimal import Decimal
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
TIMEOUT_NOTIFICATION_THRESHOLD: datetime.timedelta = datetime.timedelta(minutes=0.5)  # If a member gets timed out for more than this, member gets notified
TIMEOUT_DURATION_OUTLINE: dict[float, float] = {1.0: 0.0, 0.0: 0.0, TIMEOUT_THRESHOLD: TIMEOUT_NOTIFICATION_THRESHOLD.total_seconds() / 60.0, -1.0: 20.0, -2.0: 300.0, -3.0: 10080.0, -4.0: 10080.0}  # Score: Timeout duration (minutes)
CREDIT_THRESHOLDS = {float('-inf'): 0.0, 0.2: 0.125, 0.25: 0.1875, 0.5: 0.5, 1.0: 0.875}  # deep_score threshold: credits
REQUIRED_ROLES: list[set[int]] = [{1225900663746330795, 1225899714508226721, 1225900752225177651, 1225900807216562217, 1260753793566511174}, {1256626845970075779, 1256627378763993189},
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

# Load environment variables from .env file
load_dotenv()

# Load the token from an environment variable
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# Initialize the bot with the necessary intents
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='', intents=intents)

# Path to the data file
data_file: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data.json"))

# Configure logging, excluding discord logs
logger = logging.getLogger('kams-bot')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter(LOGGING_FORMAT)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)

# Extract x and y coordinates from the dictionary
x_coords: np.ndarray = np.array(list(TIMEOUT_DURATION_OUTLINE.keys()))
y_coords: np.ndarray = np.array(list(TIMEOUT_DURATION_OUTLINE.values()))

# Create a linear interpolation function
linear_interp = interp1d(x_coords, y_coords, fill_value='extrapolate')  # linear interpolation


# Function to evaluate the linear interpolation at any given x
def calculate_timeout(x: float) -> float:
    """
    Calculate the timeout duration based on the shallow score.
    :param x:
    :return:
    """
    return float(linear_interp(x))


# Generate points to plot the function
x_values: np.ndarray = np.linspace(min(x_coords), max(x_coords), 500)
y_values: np.ndarray = linear_interp(x_values)

# Extract the keys and values from the CREDIT_THRESHOLDS dictionary
thresholds = np.array(list(CREDIT_THRESHOLDS.keys()))
credit_allocations = np.array(list(CREDIT_THRESHOLDS.values()))

# Create an interpolation function
credit_calculation_function = interp1d(thresholds, credit_allocations, kind="previous", fill_value="extrapolate")


def compute_credits(deep_score: float) -> float:
    """
    Compute the credits based on the deep score.
    :param deep_score:
    :return:
    """
    return float(credit_calculation_function(deep_score))


# Helper function to load data from JSON file
default_user_entry: dict[str, float] = {"shallow_score": 0.0, "deep_score": 0.0, "credits": compute_credits(0.0)}

DataType = dict[int, dict[str, float]]

data_lock = asyncio.Lock()


async def load_data() -> DataType:
    """
    Load data from the JSON file.
    :return:
    """
    if not os.path.exists(data_file):
        return {}
    with open(data_file) as file:  # Read file, converting the ids to integers
        return {int(key): value for key, value in json.load(file).items()}


# Helper function to save data to JSON file
def save_data(data: DataType, output_file: str = data_file) -> None:
    """
    Save data to the JSON file.
    :param output_file:
    :param data:
    """
    # Convert the keys to strings
    converted_data = {str(key): value for key, value in data.items()}
    with open(output_file, 'w') as file:
        json.dump(converted_data, file, indent=2)


async def set_respect_role(guild: discord.Guild, member: discord.Member, score: float) -> None:
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

    if score > 0:
        if disrespectful_role in member.roles:
            await member.remove_roles(disrespectful_role, reason=f"Respect score of {score} is positive.")
        if respectful_role not in member.roles:
            await member.add_roles(respectful_role, reason=f"Respect score of {score} is positive.")
            logger.info(f"{member.display_name} has been upgraded to '{RESPECTFUL_ROLE_NAME}'.")
    elif score < min(-1.0, -0.01 * sum(not memb.bot for memb in guild.members)):
        if respectful_role in member.roles:
            await member.remove_roles(respectful_role, reason=f"Respect score of {score} is unacceptably bad.")
        if disrespectful_role not in member.roles:
            await member.add_roles(disrespectful_role)
            logger.info(f"{member.display_name} has been downgraded to '{DISRESPECTFUL_ROLE_NAME}'.")


@bot.event
async def on_message(_message: discord.Message) -> None:
    """

    :param _message:
    """
    pass


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
        await interaction.response.send_message("Select whether you would like request to ban or to unban a user.", view=select, ephemeral=True)


class SetSlowmodeModal(discord.ui.Modal, title="Set Slowmode"):
    length = discord.ui.TextInput(label="Length (seconds)", required=True)
    reset_time = discord.ui.TextInput(label="Reset Time (optional)", required=False)

    async def on_submit(self, interaction: discord.Interaction):
        # Validate input for length
        try:
            length = int(self.length.value)
            if length < 0:
                raise ValueError("Slowmode must be a non-negative integer.")
        except ValueError:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(content=f"{ERROR_SYMBOL} Invalid slowmode length. Slowmode must be a non-negative integer.")
            return

        # Optional reset time validation
        reset_time: int | None = None
        if self.reset_time.value:
            try:
                reset_time = int(self.reset_time.value)
                if reset_time <= 0:
                    raise ValueError("Reset time must be a positive number.")
            except ValueError:
                # noinspection PyUnresolvedReferences
                await interaction.response.edit_message(content=f"{ERROR_SYMBOL} Invalid reset time. Please enter a positive number.")
                return

        if length < 0:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(content=f"{ERROR_SYMBOL} Slowmode must be a non-negative integer.")
            return
        try:
            await interaction.channel.edit(slowmode_delay=length)
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(content=f"{SUCCESS_SYMBOL} Slowmode has been set to {length} second{"s" if length != 1 else ""} {f"with a reset time of {reset_time} seconds" if reset_time is not None else ''}.")
        except discord.errors.Forbidden:
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(f"{ERROR_SYMBOL} I do not have permission to set slowmode in this channel.")


BanRequestsType = dict[int, dict[int: dict[str, bool | str]]]  # user_id: {requester_id: ban_request}


class RequestBanModal(discord.ui.Modal):

    def __init__(self, ban_bool: bool, **kwargs):
        title = "Request Ban" if ban_bool else "Request Unban"
        super().__init__(title=title, **kwargs)
        self.ban_bool = ban_bool
        self.target: discord.ui.TextInput = discord.ui.TextInput(label=f"User ID to {"Ban" if self.ban_bool else "Unban"}", required=True, placeholder="012345678910111213", style=discord.TextStyle.short)
        self.reason = discord.ui.TextInput(label=f"Reason for {"Ban" if self.ban_bool else "Unban"}", style=discord.TextStyle.paragraph, required=True, min_length=40)

        # Add the text inputs to the modal
        self.add_item(self.target)
        self.add_item(self.reason)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            target_object: discord.User = await bot.fetch_user(int(self.target.value))
        except (ValueError, discord.errors.NotFound, discord.errors.HTTPException):
            # noinspection PyUnresolvedReferences
            await interaction.response.edit_message(content=f"{ERROR_SYMBOL} Invalid user ID. Please enter a valid user ID.")
            return

        reason = self.reason.value
        # Open file ban_requests.json, create it if it doesn't exist
        ban_requests: BanRequestsType = read_ban_requests()
        if target_object.id not in ban_requests:
            ban_requests[target_object.id] = {}
        ban_requests[target_object.id][interaction.user.id] = {"request": self.ban_bool, "reason": reason}
        save_ban_requests(ban_requests)
        if self.ban_bool:
            # If all current justices have requested a ban, ban the user
            justice_ids: list[int] = await get_justice_ids(interaction.guild)
            if len(justice_ids) == JUSTICE_COUNT and all(justice_id in ban_requests[target_object.id] and ban_requests[target_object.id][justice_id] for justice_id in justice_ids):
                # Ban the user
                try:
                    await interaction.guild.ban(target_object, reason="Requested by justices.")
                except discord.errors.Forbidden:
                    # noinspection PyUnresolvedReferences
                    await interaction.response.edit_message(f"{ERROR_SYMBOL} Sorry, I am unable to ban this user.")
        else:
            # If 2/3 of current justices have requested an unban, unban the user
            justice_ids: list[int] = await get_justice_ids(interaction.guild)
            if sum(1 for justice_id in justice_ids if justice_id in ban_requests[target_object.id] and not ban_requests[target_object.id][justice_id]) >= 2 * len(justice_ids) / 3 and any(
                    ban.user.id == target_object.id async for ban in interaction.guild.bans()):
                # Unban the user
                await interaction.guild.unban(target_object, reason="Requested by justices.")
        # noinspection PyUnresolvedReferences
        await interaction.response.edit_message(content=f"{SUCCESS_SYMBOL} {"Unb" if not self.ban_bool else 'B'}an request submitted for user \"{target_object.display_name}\".")


class BanUnbanSelect(discord.ui.Select):
    def __init__(self):
        # To make it a multi-select dropdown, add the parameter max_values=2
        options = [discord.SelectOption(label="Ban", value="ban"), discord.SelectOption(label="Unban", value="unban"), discord.SelectOption(label="Cancel Prior Request", value="cancel")]
        super().__init__(placeholder="Select an action", options=options)

    async def callback(self, interaction: discord.Interaction):
        # Save the user's choice in the interaction or context
        choice = self.values[0]
        if choice == "cancel":
            ban_requests: BanRequestsType = read_ban_requests()
            user_requests: list[tuple[int, str, bool]] = [(target_id, (await bot.fetch_user(interaction.user.id)).display_name, ban_requests[target_id][interaction.user.id]["request"]) for target_id in ban_requests if
                                                          interaction.user.id in ban_requests[target_id]]
            if not user_requests:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(f"{ERROR_SYMBOL} You have no ban requests to cancel.", ephemeral=True)
            else:
                select = BanRequestCancelSelect(requests=user_requests)
                view: discord.ui.View = discord.ui.View()
                view.add_item(select)
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message("Select a ban request to cancel.", view=view, ephemeral=True)
            return
        # Pass the choice to the modal
        modal = RequestBanModal(ban_bool=choice == "ban")
        # noinspection PyUnresolvedReferences
        await interaction.response.send_modal(modal)


class BanRequestCancelSelect(discord.ui.Select):
    def __init__(self, requests: list[tuple[int, str, bool]]):
        # Ban/Unban [User id] ("display name")
        options = [discord.SelectOption(label=f'{"Ban" if request[2] else "Unban"} {request[0]} ("{request[1]}")', value=str(request[0])) for request in requests]
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
                await interaction.response.send_message(f"{SUCCESS_SYMBOL} Ban request for user {target_id} (\"{next(request[1] for request in self.requests if request[0] == target_id)}\") has been cancelled.", ephemeral=True)
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
    if os.path.exists("ban_requests.json"):
        with open("ban_requests.json") as file:
            return {int(key): {int(subkey): value for subkey, value in subdict.items()} for key, subdict in json.load(file).items()}
    return {}

def save_ban_requests(ban_requests: BanRequestsType) -> None:
    """
    Save the ban requests to the JSON file.
    :param ban_requests:
    """
    with open("ban_requests.json", "w") as file:
        json.dump({str(key): {str(subkey): value for subkey, value in subdict.items()} for key, subdict in ban_requests.items()}, file, indent=2)

@bot.tree.command(name="justice_toolbox", description="Access the Justice Toolbox.")
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
        await interaction.response.send_message("You must be a Justice to access the Justice Toolbox.", ephemeral=True)
        return
    # noinspection PyUnresolvedReferences
    await interaction.response.send_message("Justice Toolbox", view=JusticeToolboxView(), ephemeral=True)


@bot.tree.command(name="vote", description="Vote for a user with a severity ranging from -1 to 1. See The Rules for more information.")
@commands.guild_only()
async def slash_vote(interaction: discord.Interaction, target: discord.User, severity: float, reason: str) -> None:
    """
    Vote for a user with a severity ranging from -1 to 1.
    :param reason:
    :param interaction:
    :param target:
    :param severity:
    :return:
    """

    if severity == 0.0:
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
        if -1.0 <= severity <= 1.0:
            if abs(severity) > data[interaction.user.id]["credits"]:
                # noinspection PyUnresolvedReferences
                await interaction.response.send_message(f"Insufficient credits! You only have {data[interaction.user.id]['credits']} credits remaining.", ephemeral=True, delete_after=15)
                logger.info(f"Insufficient credits for {interaction.user.display_name} to vote for {target.display_name} with severity {severity}.")
                return
            else:
                data[interaction.user.id]["credits"] = float(Decimal(str(data[interaction.user.id]["credits"])) - Decimal(str(abs(severity))))
        else:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message("Invalid severity value. Please use a value between -1 and 1.", ephemeral=True)
            logger.info(f"Invalid severity value for {interaction.user.display_name} to vote for {target.display_name} with severity {severity}.")
            return

        data[target.id]["shallow_score"] = float(Decimal(str(data[target.id]["shallow_score"])) + Decimal(str(severity)))
        if target_member is not None:
            await set_respect_role(interaction.guild, target_member, data[target.id]["shallow_score"] + data[target.id]["deep_score"])
            if data[target.id]["shallow_score"] < (TIMEOUT_THRESHOLD + 1.0):
                # Timeout procedure
                timeout_minutes = calculate_timeout(data[target.id]["shallow_score"] + min(data[target.id]["deep_score"], 0.5))
                old_duration: datetime.timedelta = datetime.timedelta()
                if target_member.timed_out_until is not None and (target_member.timed_out_until - discord.utils.utcnow()) > old_duration:
                    old_duration = target_member.timed_out_until - discord.utils.utcnow()
                new_duration: datetime.timedelta = datetime.timedelta(minutes=timeout_minutes)
                if (severity < 0 or new_duration < old_duration) and new_duration != old_duration:
                    until: datetime.datetime = discord.utils.utcnow() + new_duration
                    if "suspended_timeout" in data[target_member.id]:
                        data[target_member.id]["suspended_timeout"] = new_duration.total_seconds()
                    else:
                        try:
                            await target_member.edit(timed_out_until=until, reason=f"Voted {severity} by a member.")
                            logger.info(f"{target_member.display_name} has been timed out for {timeout_minutes} minutes.")
                            if old_duration < TIMEOUT_NOTIFICATION_THRESHOLD < new_duration:
                                await dm_member(target_member,
                                                f"You have been timed out for {timeout_minutes} minutes due to your low respect score. Please take this time to reflect on your behavior. If you have any questions, feel free to reach out "
                                                f"to a "
                                                f"moderator.")
                        except discord.errors.Forbidden:
                            logger.error(f"Forbidden to timeout user \"{target_member.display_name}\" (id={target_member.id}).")

        save_data(data)

        # Send a message publicly
        public_message: str = f"{interaction.user.mention} has {'up' if severity > 0 else 'down'}voted {target.mention} with severity {severity}. Reason: {reason}"
        await interaction.channel.send(public_message)

        try:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message(f"Vote successful! You have {data[interaction.user.id]['credits']} credits remaining.", ephemeral=True, delete_after=15)
        except discord.errors.NotFound:
            logger.error(f"Interaction not found to send vote confirmation to \"{interaction.user.display_name}\". Processing may have taken too long. Proceeding to send a DM.")
            await dm_member(interaction.user, f"With apologies for the delay, your vote for {target.display_name} with severity {severity} has been successfully processed. You have {data[interaction.user.id]['credits']} credits remaining.")


@bot.tree.command(name="credits", description="Check the number of credits you have.")
async def slash_credits(interaction: discord.Interaction) -> None:
    """
    Check the number of credits a user has.
    :param interaction:
    :return:
    """
    async with data_lock:
        data: DataType = await load_data()
        if interaction.user.id not in data:
            await on_member_join(interaction.user)
        try:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message(f"You have {data[interaction.user.id]['credits']} credits remaining.", ephemeral=True)
        except discord.errors.NotFound:
            logger.error(f"Interaction not found to send credit count to \"{interaction.user.display_name}\". Processing may have taken too long.")


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


is_initialized = False


@bot.event
async def on_ready() -> None:
    """
    Event that runs when the bot is ready, syncing the commands and starting the day_change loop.
    """

    global is_initialized
    if is_initialized:
        return
    async with data_lock:
        logger.info("Bot is ready, starting to sync commands...")
        await bot.tree.sync(guild=bot.get_guild(GUILD_ID))
        logger.info("Slash commands synced!")
        day_change.start()
        logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    is_initialized = True


async def get_justice_ids(guild: discord.Guild) -> list[int]:
    """
    Get the justice ids from the justices channel.

    :return:
    """
    # Fetch justice ids from the justices channel
    justice_channel_category: discord.CategoryChannel | None = discord.utils.get(guild.categories, name=JUSTICE_CHANNEL_CATEGORY)
    if justice_channel_category is not None:
        justice_channel: discord.TextChannel | None = discord.utils.get(justice_channel_category.text_channels, name=JUSTICE_CHANNEL_NAME)
        if justice_channel is not None:
            async for message in justice_channel.history(limit=1):
                if message.author == bot.user:
                    # Extract the mentions from the message
                    return [mention.id for mention in message.mentions]
    return []


@bot.event
async def on_member_join(member: discord.Member) -> None:
    """
    Event that runs when a member joins the server, welcoming them and setting their roles.
    :param member:
    :return:
    """
    # Ensure the member is not a bot
    if member.bot:
        return
    message_sent: bool = False
    async with data_lock:
        data: DataType = await load_data()
        if not member.id in data:
            # DM the member
            sending_message: str = ""
            # If the member joined over 5 minutes ago
            if (discord.utils.utcnow() - member.joined_at).total_seconds() > 300:
                sending_message += "With apologies for the delay,\n"
            sending_message += ("Welcome to the KaMS Club Discord server! As you may have noticed in the rules, your nickname must include your real-life name. Please make sure to update your nickname accordingly if you haven't already. "
                                "Thanks!")
            await dm_member(member, sending_message)
            message_sent = True

            data[member.id] = default_user_entry
            save_data(data)
        else:
            await set_justice_role(member, await get_justice_ids(member.guild))

        await set_respect_role(member.guild, member, data[member.id]["shallow_score"] + data[member.id]["deep_score"])
    logger.info(f"{member.display_name} has been welcomed to the server {"(no message was sent because this isn't their first time) " if not message_sent else ""}and their roles have been set.")


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


def justice_score(data: DataType, member: discord.Member) -> tuple[float, datetime.datetime]:
    """
    Calculate the justice score of a member.
    :param data:
    :param member:
    :return:
    """
    return data[member.id]["deep_score"], member.joined_at


async def delete_timeout_log(target_member_id: int, guild: discord.Guild) -> None:
    """
    Delete the timeout log message in the logger channel.
    :param guild:
    :param target_member_id:
    :return:
    """

    logger_channel: discord.TextChannel | None = discord.utils.get(guild.text_channels, name=LOGGER_CHANNEL_NAME)
    if logger_channel is not None:
        async for message in logger_channel.history(limit=100):
            if message.author.id == ELARA_LOGGER_ID and message.mentions[0].id == target_member_id and message.mentions[1].id == bot.user.id and ROLE_TIMEOUT_REASON in message.content:
                await message.delete()
                return


@tasks.loop(time=DAY_CHANGE_TIME)
async def day_change() -> None:
    """
    Loop that runs every day at midnight to update the data and assign roles.
    :return:
    """
    logger.info("Day change has started.")
    async with (data_lock):
        data: DataType = await load_data()
        backup_file_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_backup"))
        if not os.path.exists(backup_file_path):
            os.makedirs(backup_file_path)
        save_data(data, backup_file_path + f"/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")  # Backup data

        guild: discord.Guild | None = bot.get_guild(GUILD_ID)
        if guild is None:
            logger.error("Could not find provided guild.")
            return
        member_count: int = 0
        for member in guild.members:
            if not member.bot:
                member_count += 1
            if member.id not in data:
                await on_member_join(member)
        for member_id in data:
            if data[member_id]["shallow_score"] > 0:
                data[member_id]["deep_score"] += math.sqrt(data[member_id]["shallow_score"]) / (member_count ** (1 / 3))
                data[member_id]["shallow_score"] = 0.0
            elif data[member_id]["shallow_score"] < 0:
                data[member_id]["deep_score"] += data[member_id]["shallow_score"]
                data[member_id]["shallow_score"] /= 4.0
                if data[member_id]["shallow_score"] > -0.01:
                    data[member_id]["shallow_score"] = 0.0
            elif data[member_id]["deep_score"] > 0.1:
                data[member_id]["deep_score"] -= 0.0078125  # 1/28

        # Calculate justices
        justices: list[discord.Member] = []
        if len(data.keys()) >= JUSTICE_COUNT * 5:
            justices = sorted(guild.members, key=lambda memb: justice_score(data, memb), reverse=True)[:JUSTICE_COUNT]
            if data[justices[-1].id]["deep_score"] <= JUSTICE_DEEP_SCORE_REQUIREMENT:
                justices = []

        for member_id in data:
            member = guild.get_member(member_id)

            if member is not None:
                await set_justice_role(member, [j.id for j in justices])
            data[member_id]["credits"] = compute_credits(data[member_id]["deep_score"])

            # Timeout members that are missing required roles
            if member is not None and not member.bot:
                member_role_ids: set[int] = set(rl.id for rl in member.roles)
                if not all(member_role_ids & role_category for role_category in REQUIRED_ROLES):
                    try:
                        was_timed_out: bool = member.timed_out_until is not None and member.timed_out_until > discord.utils.utcnow()
                        await member.timeout(MISSING_ROLE_TIMEOUT_DURATION, reason="Missing required roles.")
                        if "suspended_timeout" not in data[member_id]:
                            data[member_id]["suspended_timeout"] = 0.0 if not was_timed_out else max(0.0, (member.timed_out_until - discord.utils.utcnow()).total_seconds())
                            await dm_member(member, MISSING_ROLE_MESSAGE(was_timed_out))
                            logger.info(f"{member.display_name} (id={member_id}) has been timed out for {MISSING_ROLE_TIMEOUT_DURATION.total_seconds() / 86400.0} days due to missing required roles.")
                        else:
                            asyncio.get_event_loop().call_later(30, delete_timeout_log, member.id, guild)
                    except discord.errors.Forbidden:
                        logger.error(f"Forbidden to timeout user \"{member.display_name}\" (id={member_id}) for missing required roles.")
        # If less than 10% of members have any credits, give everyone exactly 0.1 credits
        if sum(1 for member_id in data if data[member_id]["credits"] > 0.0) < 0.1 * len(data):
            for member_id in data:
                data[member_id]["credits"] = 0.1

        save_data(data)
    logger.info("Data update complete.")

    # Make a leaderboard of the five justices
    message_content: str = ""
    i: int
    for i, justice_member in enumerate(justices):
        data[justice_member.id]["credits"] += (JUSTICE_COUNT - i) * 0.5
        message_content += f"{i + 1}. {justice_member.mention}\n"
    if not message_content:
        message_content = "No justices have been determined yet."

    justice_channel_category: discord.CategoryChannel | None = discord.utils.get(guild.categories, name=JUSTICE_CHANNEL_CATEGORY)
    if justice_channel_category is None:
        justice_channel_category = await guild.create_category(JUSTICE_CHANNEL_CATEGORY)
        assert justice_channel_category is not None
    justice_channel: discord.TextChannel | None = discord.utils.get(justice_channel_category.text_channels, name=JUSTICE_CHANNEL_NAME)
    if justice_channel is None:
        justice_channel = await justice_channel_category.create_text_channel(JUSTICE_CHANNEL_NAME, overwrites={guild.default_role: discord.PermissionOverwrite(send_messages=False, create_public_threads=False, create_private_threads=False),
                                                                                                               discord.utils.get(guild.roles, name="KaMS Club"): discord.PermissionOverwrite(send_messages=True)})
    else:
        async for message in justice_channel.history():
            await message.delete()
    await justice_channel.send(message_content, allowed_mentions=discord.AllowedMentions.none())

    # Cleanup polls channel
    deleted_count: int = 0
    for channel in guild.text_channels:
        if channel.name == "polls":
            # after one week ago
            async for message in channel.history(after=discord.utils.utcnow() - datetime.timedelta(days=7)):
                if message.poll is None and not message.pinned and not message.content.startswith("[POLL]"):
                    await message.delete()
                    deleted_count += 1
    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} non-poll messages in the polls channel.")
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
    if not all(set(rl.id for rl in before.roles) & role_category for role_category in REQUIRED_ROLES) and all(set(rl.id for rl in after.roles) & role_category for role_category in REQUIRED_ROLES):
        assert before.id == after.id
        async with data_lock:
            data: DataType = await load_data()
            if not after.id in data:
                await on_member_join(after)
            if "suspended_timeout" in data[after.id]:
                try:
                    if data[after.id]["suspended_timeout"] > 0.0:
                        await after.timeout(datetime.timedelta(seconds=data[after.id]["suspended_timeout"]), reason="Resume timeout from before role-acquisition obligation.")
                    else:
                        await after.timeout(None, reason="Acquired necessary roles.")
                except discord.errors.Forbidden:
                    logger.error(f"Forbidden to untimeout user \"{after.display_name}\" (id={after.id}) for role acquisition.")
                    return
                logger.info(f"{after.display_name} (id={after.id}) has been untimed out due to acquiring the necessary roles.")
                if after.timed_out_until is not None and (after.timed_out_until - discord.utils.utcnow()) > TIMEOUT_NOTIFICATION_THRESHOLD:
                    logger.info(f"{after.display_name} (id={after.id}) still has a respect timeout of {data[after.id]['suspended_timeout'] / 60.0} minutes to serve.")
                    await dm_member(after, f"Your role timeout has been removed, but you still have a timeout of {data[after.id]['suspended_timeout'] / 60.0} minutes to serve.")
                else:
                    await dm_member(after, ROLE_RESTORATION_MESSAGE)
                del data[after.id]["suspended_timeout"]
                save_data(data)


async def shutdown() -> None:
    """
    Gracefully shuts down the bot.
    """
    logger.info("Waiting to acquire data_lock...")
    await data_lock.acquire()
    logger.info("Data lock acquired; closing bot...")
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

    # Optional: delay before restarting
    time.sleep(2)

    # Restart the script
    print("Restarting script...")
    os.execv(sys.executable, ['python3.12', __file__])


# Register the global exception handler
sys.excepthook = handle_exception

# Run the bot
bot.run(TOKEN, log_handler=console_handler, log_formatter=formatter)
