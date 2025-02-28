#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <bearer_token> <query> <session_id> <request_id>"
    exit 1
fi
BEARER="$1"
QUERY="$2"
SESSION_ID="$3"
REQUEST_ID="$4"
PORT=8001
USER=cb
curl -X POST "http://localhost:$PORT/api/thirdbrain-mcp-openai-agent" \
    -H "Authorization: Bearer $BEARER" \
    -H "Content-Type: application/json" \
    -d "{
        \"query\": \"$QUERY\",
        \"user_id\": \"$USER\",
        \"request_id\": \"R123_$REQUEST_ID\",
        \"session_id\": \"$SESSION_ID\"
    }"
