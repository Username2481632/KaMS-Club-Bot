#!/bin/bash

jq '
  # Recursively update all user objects across all guilds
  map_values(
    map_values(
      . * {
        credibility: "0",
        latest_message_time: 1420070400.0,
        conversation_start_time: 1420070400.0,
        suspended_timeout: null
      }
    )
  )' data.json > data.tmp && mv data.tmp data.json