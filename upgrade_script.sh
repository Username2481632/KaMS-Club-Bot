#!/bin/bash

# Define the static required_roles array
jq -r --argjson roles '[
  [
    1225900663746330795,
    1225899714508226721,
    1225900752225177651,
    1225900807216562217,
    1260753793566511174
  ],
  [
    1256626845970075779,
    1256627378763993189
  ],
  [
    1261372426382737610,
    1261371054161662044
  ]
]' '.["1219413083504644257"] += { required_roles: $roles }' config.json > config.tmp && mv config.tmp config.json