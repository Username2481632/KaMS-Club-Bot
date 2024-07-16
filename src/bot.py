import datetime
import json
import os
import signal
import sys
import discord
import numpy as np
from discord.ext import commands
from dotenv import load_dotenv
from scipy.optimize import least_squares

# Load environment variables from .env file
load_dotenv()

# Load the token from an environment variable
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# Initialize the bot with the necessary intents
intents = discord.Intents.default()
intents.members = True  # Enable the members intent
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Path to the data file
data_file = '../data.json'

# Cooldown duration (24 hours in seconds)
COOLDOWN_DURATION = 24 * 60 * 60

# Timeout duration function
desired_points = {0.1: 2, 1: 10080}


# Define the function to fit
def fit_function(x, L, k):
    return L / (1 + np.exp(-k * (x - 0.5)))


# Define the cost function for least_squares
def cost_function(params, local_x_data, local_y_data):
    return fit_function(local_x_data, *params) - local_y_data


# Convert desired_points dictionary to numpy arrays
x_data = np.array(list(desired_points.keys()))
y_data = np.array(list(desired_points.values()))

# Use least_squares to find the best parameters
result = least_squares(cost_function, [1, 1], args=(x_data, y_data))

# Extract the optimal values of L and k
L_opt, k_opt = result.x


# Function to calculate timeout, using the original y_data values
def calculate_timeout(score):
    return fit_function(score, L_opt, k_opt)


# Helper function to load data from JSON file
default_user_entry = {"score": 0.0, "voting_times": {}}
def load_data(*member_ids):
    if not os.path.exists(data_file):
        return {member_id: default_user_entry for member_id in member_ids}
    with open(data_file, 'r') as file:
        data = json.load(file)
        for member_id in member_ids:
            if member_id not in data:
                data[member_id] = default_user_entry
        return data



# Helper function to save data to JSON file
def save_data(data):
    with open(data_file, 'w') as file:
        json.dump(data, file, indent=4)


async def set_respect_role(guild: discord.Interaction.guild, member: discord.Member, score: float, total_members = None):
    disrespectful_role: discord.Role = discord.utils.get(guild.roles, name="Disrespectful :(")
    respectful_role: discord.Role = discord.utils.get(guild.roles, name="Respectful :)")

    if disrespectful_role is None or respectful_role is None:
        print("The 'Disrespectful :(' or 'Respectful :)' role does not exist.")
        return

    if score > 1.0:
        if respectful_role in member.roles:
            await member.remove_roles(respectful_role)
            await member.add_roles(disrespectful_role)
            print(f"{member.display_name} has been downgraded to 'Disrespectful :('.")
    else:
        if not total_members:
            total_members = sum(not member.bot for member in guild.members)
        if score < min(-5.0, 0.1 * total_members):
            if respectful_role in member.roles:
                await member.remove_roles(respectful_role)
                await member.add_roles(disrespectful_role)
                print(f"{member.display_name} has been downgraded to 'Disrespectful :('.")
        else:
            if disrespectful_role in member.roles:
                await member.remove_roles(disrespectful_role)
                await member.add_roles(respectful_role)
                print(f"{member.display_name} has been upgraded to 'Respectful :)'.")
    return


# Vote function for both slash and regular commands
async def vote(ctx: discord.Interaction, target: discord.User, action: str):
    assert action in ["upvote", "downvote"]
    user_id = str(ctx.user.id)
    target_id = str(target.id)
    data = load_data(user_id, target_id)

    if target_id not in data[user_id]['voting_times'] or datetime.datetime.now().timestamp() - data[user_id]["voting_times"][target_id] > COOLDOWN_DURATION:
        if action == 'upvote':
            data[target_id]["score"] += 1.0
            if data[target_id]["score"] > 1.0:
                target_member = ctx.guild.get_member(target.id)
                if target_member:
                    await set_respect_role(ctx.guild, target_member, data[target_id]["score"])
        else:
            if data[user_id]["score"] > 10.0:
                data[target_id]["score"] = min(data[target_id]["score"] / 2.0, data[target_id]["score"] - 1.0)
            else:
                data[target_id]["score"] -= 1.0
            total_members = sum(not member.bot for member in ctx.guild.members)
            if data[target_id]["score"] < min(-5.0, 0.1 * total_members):
                target_member = ctx.guild.get_member(target.id)  # Convert User to Member
                if target_member:  # Check if the member exists in the guild
                    await set_respect_role(ctx.guild, target_member, data[target_id]["score"], total_members)
                    await timeout_user(target_member, calculate_timeout((-data[user_id]["score"]) / total_members))
        data[user_id]["voting_times"][target_id] = datetime.datetime.now().timestamp()
    else:
        remaining_time = COOLDOWN_DURATION - (datetime.datetime.now().timestamp() - data[user_id]["voting_times"][target_id])
        remaining_hours = int(remaining_time // 3600)
        remaining_minutes = int((remaining_time % 3600) // 60)
        remaining_seconds = int(remaining_time % 60)
        remaining_formatted = f"{remaining_hours:02}:{remaining_minutes:02}:{remaining_seconds:02}"
        print(f"Cooldown prevented voting from {ctx.user.display_name} for {target.display_name}. Cooldown remaining: {remaining_formatted}")

    save_data(data)

    await ctx.response().send_message(f"Thank you for your feedback, {ctx.user.mention}!", ephemeral=True)
    # Send an ephemeral message to the user who was upvoted or downvoted
    if action == 'upvote':
        await target.send(f"You have received an upvote from {ctx.user.display_name}.")
    else:
        await target.send(f"You have received a downvote from {ctx.user.display_name}.")


# Define slash commands using the bot's built-in tree
@bot.tree.command(name="upvote", description="Upvote a user.")
async def slash_upvote(interaction: discord.Interaction, user: discord.User):
    await vote(interaction, user, 'upvote')


@bot.tree.command(name="downvote", description="Downvote a user.")
async def slash_downvote(interaction: discord.Interaction, user: discord.User):
    await vote(interaction, user, 'downvote')


async def timeout_user(member: discord.Member, minutes: float):
    duration = datetime.timedelta(minutes=minutes)
    await member.edit(timed_out_until=discord.utils.utcnow() + duration)
    print(f"{member.display_name} has been timed out for {minutes} minutes.")


@bot.event
async def on_ready():
    await bot.tree.sync(guild=discord.Object(id=1201368154174144602))  # Replace with your guild ID
    print("Slash commands synced!")
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")


@bot.event
async def on_member_join(member):
    # Ensure the member is not a bot
    if member.bot:
        return

    # DM the member
    try:
        await member.send("Welcome to the KaMS Club Discord server! As you may have noticed in the rules, your nickname must include your real-life name. Please make sure to update your nickname accordingly if you haven't already. Thanks!")
    except Exception as e:
        print(f"Failed to send DM to {member.display_name}: {e}")
    data = load_data()
    if not member.id in data:
        data[member.id] = default_user_entry
        save_data(data)

    await set_respect_role(member.guild, member, data[member.id]["score"])


# Define the signal handler function
def signal_handler(sig, frame) -> None:
    print('SIGTERM received. Gracefully shutting down the bot.')
    # Perform any necessary cleanup here
    bot.close()  # Gracefully close the Discord bot connection
    sys.exit(0)  # Exit the script


# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)

# Run the bot
bot.run(TOKEN)
