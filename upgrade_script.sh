#!/bin/bash

# Ensure jq is installed
if ! command -v jq &> /dev/null
then
    # Install jq
    echo "jq could not be found. Installing jq..."
    echo "koshkamyshka" | sudo -S apt-get update
    sudo apt-get install jq --yes
fi

# Define the input file
input_file="../data.json"

# If the file does not exist, exit
if [ ! -f "$input_file" ]; then
    echo "File $input_file not found."
    exit 0
fi
# Use jq to transform the data and overwrite the original file
jq 'with_entries(
    .value |= {
        shallow_score: .score,
        deep_score: 0.0,
        credits: 2.0
    } | del(.voting_times, .score)
)' "$input_file" > tmp.$$.json && mv tmp.$$.json "$input_file"

echo "Transformation complete. Data saved to $input_file."