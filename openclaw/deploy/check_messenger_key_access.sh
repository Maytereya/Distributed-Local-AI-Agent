#!/bin/sh
# Check mounted credential access as each service's actual runtime user.
# Run on the deployment host; no key/header contents are printed.
set -eu
for name in agent-api support-messenger-aggregator-web-1 support-messenger-aggregator-telegram_bot-1; do
    printf '%s uid=' "$name"
    docker exec "$name" id -u
    docker exec "$name" sh -c 'test -r /run/secrets/messenger_api_key'
    docker exec "$name" stat -c 'key_file_metadata=%u:%g:%a' /run/secrets/messenger_api_key
done
