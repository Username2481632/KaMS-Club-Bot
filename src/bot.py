import asyncio
import datetime
import json
import logging
import math
import os
import shutil
import signal
from decimal import Decimal

import aiocron
import discord
import numpy as np
from discord.ext import commands
from dotenv import load_dotenv
from scipy.interpolate import interp1d

# Parameters
MEMBER_CREDITS: float = 1.5
GUILD_ID: int = 1201368154174144602
JUSTICE_COUNT: int = 5
JUSTICE_CHANNEL_NAME: str = "justices"
JUSTICE_CHANNEL_CATEGORY: str = "Information"
TIMEOUT_THRESHOLD: float = -0.5  # If a member's shallow score falls below this value, member gets timed out
TIMEOUT_NOTIFICATION_THRESHOLD: datetime.timedelta = datetime.timedelta(minutes=0.5)  # If a member gets timed out for more than this, member gets notified
TIMEOUT_DURATION_OUTLINE: dict[float, float] = {1.0: 0.0, 0.0: 0.0, TIMEOUT_THRESHOLD: TIMEOUT_NOTIFICATION_THRESHOLD.total_seconds() / 60.0, -1.0: 10.0, -2.0: 300.0, -3.0: 10080.0, -4.0: 10080.0}  # Score: Timeout duration (minutes)
CREDIT_THRESHOLD: float = 1.0  # Deep score threshold above which a member receives full credits
MIN_CREDITS: float = 0.75  # Number of credits given to members under the CREDIT_THRESHOLD
REQUIRED_ROLES: list[set[int]] = [{1225900663746330795, 1225899714508226721, 1225900752225177651, 1225900807216562217, 1260753793566511174}, {1261372426382737610, 1261371054161662044},
                                  {1256626845970075779, 1256627378763993189}]  # Ids of roles that are required to access the server
MISSING_ROLE_MESSAGE = ("Hi there. It seems like you're missing some roles, which is why you've been temporarily timed out. No worries, though! To regain access to the server, just visit the <id:customize> tab to assign yourself the "
                        "necessary roles. If you have any questions or need assistance, feel free to reach out to a moderator. We're here to help!")
ROLE_RESTORATION_MESSAGE = "You have been untimed out due to acquiring the necessary roles. Welcome back!"
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
MISSING_ROLE_TIMEOUT_DURATION: datetime.timedelta = datetime.timedelta(days=2)

# Load environment variables from .env file
load_dotenv()

# Load the token from an environment variable
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# Initialize the bot with the necessary intents
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='/', intents=intents)

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
linear_interp = interp1d(x_coords, y_coords, kind='linear', fill_value='extrapolate')


# Function to evaluate the linear interpolation at any given x
def calculate_timeout(x: float) -> float:
    return float(linear_interp(x))


# Generate points to plot the function
x_values: np.ndarray = np.linspace(min(x_coords), max(x_coords), 500)
y_values: np.ndarray = linear_interp(x_values)

# Helper function to load data from JSON file
default_user_entry: dict[str, str | float] = {"shallow_score": 0.0, "deep_score": 0.0, "credits": MIN_CREDITS}

DataType = dict[str, dict[str, float]]

data_lock = asyncio.Lock()


async def load_data(*member_ids: str) -> DataType:
    await data_lock.acquire()
    if not os.path.exists(data_file):
        return {member_id: default_user_entry for member_id in member_ids}
    with open(data_file, 'r') as file:
        data: DataType = json.load(file)
        for member_id in member_ids:
            if member_id not in data:
                data[member_id] = default_user_entry
        return data


# Helper function to save data to JSON file
def save_data(data: DataType) -> None:
    with open(data_file, 'w') as file:
        json.dump(data, file, indent=4)
    data_lock.release()


async def set_respect_role(guild: discord.Guild, member: discord.Member, score: float) -> None:
    disrespectful_role: discord.Role | None = discord.utils.get(guild.roles, name="Disrespectful :(")
    respectful_role: discord.Role | None = discord.utils.get(guild.roles, name="Respectful :)")

    if disrespectful_role is None or respectful_role is None:
        logger.error("The 'Disrespectful :(' or 'Respectful :)' role does not exist.")
        return

    if score > 0:
        if disrespectful_role in member.roles:
            await member.remove_roles(disrespectful_role)
            await member.add_roles(respectful_role)
            logger.info(f"{member.display_name} has been upgraded to 'Respectful :)'.")
    elif score < min(-1.0, -0.01 * sum(not memb.bot for memb in guild.members)):
        if respectful_role in member.roles:
            await member.remove_roles(respectful_role)
            await member.add_roles(disrespectful_role)
            logger.info(f"{member.display_name} has been downgraded to 'Disrespectful :('.")


@bot.tree.command(name="vote", description="Vote for a user with a severity ranging from -1 to 1. See The Rules for more information.")
async def slash_vote(interaction: discord.Interaction, target: discord.User, severity: float):
    # Make sure this isn't a dm
    if interaction.guild is None:
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
        return
    if severity == 0.0:
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message("You cannot vote with a severity of 0.", ephemeral=True)
        return
    user_id: str = str(interaction.user.id)
    target_id: str = str(target.id)
    data: DataType = await load_data(user_id, target_id)
    if -1.0 <= severity <= 1.0:
        if abs(severity) > data[user_id]["credits"]:
            # noinspection PyUnresolvedReferences
            await interaction.response.send_message(f"Insufficient credits! You only have {data[user_id]['credits']} credits remaining.", ephemeral=True, delete_after=10)
            logger.info(f"Insufficient credits for {interaction.user.display_name} to vote for {target.display_name} with severity {severity}.")
            return
        else:
            data[user_id]["credits"] = float(Decimal(str(data[user_id]["credits"])) - Decimal(str(abs(severity))))
    else:
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message("Invalid severity value. Please use a value between -1 and 1.", ephemeral=True)
        logger.info(f"Invalid severity value for {interaction.user.display_name} to vote for {target.display_name} with severity {severity}.")
        return

    data[target_id]["shallow_score"] = float(Decimal(str(data[target_id]["shallow_score"])) + Decimal(str(severity)))
    target_member: discord.Member | None = interaction.guild.get_member(target.id)
    if target_member is not None:
        await set_respect_role(interaction.guild, target_member, data[target_id]["shallow_score"])
        if data[target_id]["shallow_score"] < (TIMEOUT_THRESHOLD + 1.0):
            await timeout_member(target_member, calculate_timeout(data[target_id]["shallow_score"]), data)

    save_data(data)

    # Send a message publicly
    public_message = f"{interaction.user.mention} has {'up' if severity > 0 else 'down'}voted {target.mention} with severity {severity}."
    # async for previous_message in interaction.channel.history(limit=1):
    #     if previous_message.author == bot.user:
    #         await previous_message.edit(content=f"{previous_message.content}\n{public_message}")
    #         break
    # else:
    await interaction.channel.send(public_message)

    try:
        # noinspection PyUnresolvedReferences
        await interaction.response.send_message(f"Vote successful! You have {data[user_id]['credits']} credits remaining.", ephemeral=True, delete_after=10)
    except discord.errors.NotFound:
        logger.error(f"Interaction not found to send vote confirmation to \"{interaction.user.display_name}\". Processing may have taken too long.")


async def dm_member(member: discord.Member, message: str) -> None:
    if member.dm_channel is None:
        await member.create_dm()
    try:
        await member.send(message)
    except discord.errors.Forbidden:
        logger.error(f"Forbidden to send message to \"{member.display_name}\" (id={member.id}).")


async def timeout_member(member: discord.Member, minutes: float, data: DataType) -> None:
    new_duration: datetime.timedelta = datetime.timedelta(minutes=minutes)
    old_duration: datetime.timedelta = member.timed_out_until - discord.utils.utcnow() if member.timed_out_until else datetime.timedelta(0)
    until: datetime.datetime = discord.utils.utcnow() + new_duration
    if "suspended_timeout" in data[str(member.id)]:
        data[str(member.id)]["suspended_timeout"] = new_duration.total_seconds()
    else:
        try:
            await member.edit(timed_out_until=until)
            logger.info(f"{member.display_name} has been timed out for {minutes} minutes.")
            if old_duration < TIMEOUT_NOTIFICATION_THRESHOLD < new_duration:
                await dm_member(member, f"You have been timed out for {minutes} minutes due to your low respect score. Please take this time to reflect on your behavior. If you have any questions, feel free to reach out to a moderator.")
        except discord.errors.Forbidden:
            logger.error(f"Forbidden to timeout user \"{member.display_name}\" (id={member.id}).")


@bot.event
async def on_ready():
    logger.info("Bot is ready, starting to sync commands...")
    await bot.tree.sync()
    logger.info("Slash commands synced!")
    logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")


@bot.event
async def on_member_join(member: discord.Member):
    # Ensure the member is not a bot
    if member.bot:
        return
    member_id: str = str(member.id)
    # DM the member
    await dm_member(member,
                    "Welcome to the KaMS Club Discord server! As you may have noticed in the rules, your nickname must include your real-life name. Please make sure to update your nickname accordingly if you haven't already. Thanks!")
    data: DataType = await load_data()
    if not member.id in data:
        data[member_id] = default_user_entry
        save_data(data)
    else:
        justice_ids: list[str] = calculate_justices(data)
        await set_justice_role(member, justice_ids)

    await set_respect_role(member.guild, member, data[member_id]["shallow_score"])


async def set_justice_role(member: discord.Member, justice_ids: list[str]) -> None:
    # If the Justice role does not exist, log the error and timestamp then return
    justice_role: discord.Role | None = discord.utils.get(member.guild.roles, name="Justice")
    if justice_role is None:
        logger.error("The 'Justice' role does not exist in guild \"{member.guild.name}\".")
        return
    if member.id in justice_ids and not any(role.name == "Justice" for role in member.roles):
        await member.add_roles(justice_role)
    elif member.id not in justice_ids and any(role.name == "Justice" for role in member.roles):
        await member.remove_roles(justice_role)


def calculate_justices(data: dict[str, dict[str, str | float]]) -> list[str]:
    if len(data.keys()) < JUSTICE_COUNT * 5:
        return []
    top = sorted(data.keys(), key=lambda x: data[x]["deep_score"], reverse=True)[:JUSTICE_COUNT]
    if data[top[-1]]["deep_score"] > CREDIT_THRESHOLD:
        return top
    return []


@aiocron.crontab('0 0 * * *')
async def day_change() -> None:
    data: DataType = await load_data()
    # Backup data
    if not os.path.exists("../data_backup"):
        os.makedirs("../data_backup")
    with open(f"../data_backup/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", 'w') as file:
        json.dump(data, file, indent=4)

    guild: discord.Guild | None = bot.get_guild(GUILD_ID)
    if guild is None:
        logger.error("Could not find provided guild.")
        return
    member_count: int = sum(not member.bot for member in bot.get_guild(GUILD_ID).members)
    for member_id in data:
        if data[member_id]["shallow_score"] > 0:
            data[member_id]["deep_score"] += max(0.0, math.log((data[member_id]["shallow_score"] + 0.05), member_count) + 0.7)
            data[member_id]["shallow_score"] = 0
        elif data[member_id]["shallow_score"] < 0:
            data[member_id]["deep_score"] += data[member_id]["shallow_score"]
        elif data[member_id]["deep_score"] > 0.1:
            data[member_id]["deep_score"] -= 0.1
    justice_ids: list[str] = calculate_justices(data)
    for member_id in data:
        member: discord.Member | None = guild.get_member(int(member_id))

        if member is not None:
            await set_justice_role(member, justice_ids)
        if data[member_id]["deep_score"] < CREDIT_THRESHOLD:
            data[member_id]["credits"] = MIN_CREDITS
        else:
            data[member_id]["credits"] = MEMBER_CREDITS

        # Timeout members that are missing required roles
        if not member.bot and member is not None:
            member_role_ids: set[int] = set(rl.id for rl in member.roles)
            if not all(member_role_ids & role_category for role_category in REQUIRED_ROLES):
                if "suspended_timeout" not in data[member_id]:
                    data[member_id]["suspended_timeout"] = 0.0 if member.timed_out_until is None else (member.timed_out_until - discord.utils.utcnow()).total_seconds()
                    await dm_member(member, MISSING_ROLE_MESSAGE)
                    logger.info(f"{member.display_name} (id={member_id}) has been timed out for {MISSING_ROLE_TIMEOUT_DURATION.total_seconds() / 86400.0} days due to missing required roles.")
                member.timed_out_until = discord.utils.utcnow() + MISSING_ROLE_TIMEOUT_DURATION

    # Make a leaderboard of the five justices
    message_content: str = ""
    i: int
    for i, justice_id in enumerate(justice_ids):
        justice = bot.get_user(int(justice_id))
        data[justice_id]["credits"] += (JUSTICE_COUNT - i) * 0.5
        message_content += f"{i + 1}. {justice.display_name}\n"
    if not message_content:
        message_content = "No justices have been determined yet."

    justice_channel_category: discord.CategoryChannel | None = discord.utils.get(guild.categories, name=JUSTICE_CHANNEL_CATEGORY)
    if justice_channel_category is None:
        justice_channel_category = await guild.create_category(JUSTICE_CHANNEL_CATEGORY)
    justice_channel: discord.TextChannel | None = discord.utils.get(justice_channel_category.text_channels, name=JUSTICE_CHANNEL_NAME)
    if justice_channel is None:
        justice_channel = await guild.create_text_channel(JUSTICE_CHANNEL_NAME, category=justice_channel_category)
    else:
        async for message in justice_channel.history():
            await message.delete()
    await justice_channel.send(message_content)

    save_data(data)

    # Cleanup polls channel
    for channel in guild.text_channels:
        if channel.name == "polls":
            async for message in channel.history(after=datetime.datetime(2024, 7, 21)):
                if message.poll is None and not message.pinned and not message.content.startswith("[POLL]"):
                    await message.delete()


# When a user updates their roles, check if they have the required roles
@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    if before.roles == after.roles:
        return
    if not all(set(rl.id for rl in before.roles) & role_category for role_category in REQUIRED_ROLES) and all(set(rl.id for rl in after.roles) & role_category for role_category in REQUIRED_ROLES):
        data: DataType = await load_data()
        member_id_string: str = str(after.id)
        if "suspended_timeout" in data[member_id_string]:
            try:
                if data[member_id_string]["suspended_timeout"] > 0:
                    after.timed_out_until = discord.utils.utcnow() + datetime.timedelta(seconds=data[member_id_string]["suspended_timeout"])
                else:
                    after.timed_out_until = None
            except discord.errors.Forbidden:
                logger.error(f"Forbidden to untimeout user \"{after.display_name}\" (id={after.id}) for role acquisition.")
                data_lock.release()
                return
            logger.info(f"{after.display_name} (id={after.id}) has been untimed out due to acquiring the necessary roles.")
            if after.timed_out_until is not None:
                logger.info(f"{after.display_name} (id={after.id}) still has a respect timeout of {data[member_id_string]['suspended_timeout'] / 60.0} minutes to serve.")
                await dm_member(after, f"Your role timeout has been removed, but you still have a timeout of {data[member_id_string]['suspended_timeout'] / 60.0} minutes to serve.")
            else:
                await dm_member(after, ROLE_RESTORATION_MESSAGE)
            del data[str(after.id)]["suspended_timeout"]
            save_data(data)
        else:
            data_lock.release()


async def shutdown():
    print('SIGTERM Received—Shutting Down'.center(shutil.get_terminal_size().columns))
    # Perform any necessary cleanup here
    await bot.close()  # Gracefully close the Discord bot connection


def signal_handler(sig, frame):
    loop = asyncio.get_event_loop()
    loop.create_task(shutdown())


# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)

# Run the bot
bot.run(TOKEN, log_handler=console_handler, log_formatter=formatter)
