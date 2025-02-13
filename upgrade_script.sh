#!/bin/bash
# Check if the file exists
if [ ! -f data.json ]; then
    echo "File data.json not found!"
    exit 1
fi
# Read the content of the file
content=$(cat data.json)
# Check if the content is empty
if [ -z "$content" ]; then
    echo "File data.json is empty!"
    exit 1
fi
# Add the wrapper
wrapped_content="{\"1201368154174144602\": $content}"
if echo "$wrapped_content" > data.json; then
    echo "File data.json has been successfully updated!"
else
    echo "Failed to update file data.json!"
    exit 1
fi


# Create the following file: `config.json` with the following content:
 #{
 #  "guilds": [
 #    {
 #      "id": 1219413083504644257,
 #      "welcome_dm": "Welcome to the KaMS Club Discord server! As you may have noticed in the rules, your nickname should include your real-life name. Please make sure to update your nickname accordingly if you haven't already. Thanks!",
 #      "purge_polls": true,
 #      "logger": {
 #        "enabled": true,
 #        "channel_name": "logger"
 #      }
 #    }
 #  ]
 #}
# Check if the file already exists
if [ -f config.json ]; then
    echo "File config.json already exists!"
    exit 1
fi
# Create the file with the specified content. Use multi-line string for better readability
cat <<EOL > config.json
{
  "guilds": [
    {
      "id": 1219413083504644257,
      "welcome_dm": "Welcome to the KaMS Club Discord server! As you may have noticed in the rules, your nickname should include your real-life name. Please make sure to update your nickname accordingly if you haven't already. Thanks!",
      "purge_polls": true,
      "logger": {
        "enabled": true,
        "channel_name": "logger"
      }
    }
  ]
}
EOL