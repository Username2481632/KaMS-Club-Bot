#!/bin/bash

# Use jq to transform the JSON structure and overwrite the original file
jq '.guilds | map( { ( .id | tostring ): del(.id) } ) | add' config.json > config.tmp && mv config.tmp config.json