#!/bin/bash

# Configuration
WATCH_DIR="$HOME/Downloads"
REMOTE_PATH="s_01jgsqrd7tefatat8ghx7wwhte@ssh.lightning.ai:./content/youtube_rag/backend/cookies.txt"
NGROK_URL="https://5000-01jgsqrd7tefatat8ghx7wwhte.cloudspaces.litng.ai"

# Function to wake up server and wait for response
wake_server() {
    echo "Waking up server (this may take up to 5 minutes)..."
    
    response=$(curl -X POST "${NGROK_URL}/api/chats" \
         -H "Content-Type: application/json" \
         -d "{\"title\": \"New Chat\", \"username\": \"tacotacotaco\"}" \
         -w "%{http_code}" \
         --max-time 300 \  # 5 minute timeout
         --silent \
         --output /dev/null)
    
    if [[ "$response" == *"200"* ]] || [[ "$response" == *"201"* ]]; then
        echo "Server is responsive"
        return 0
    fi
    
    echo "Server failed to respond"
    return 1
}

# Function to handle file upload
handle_cookies_file() {
    local cookies_file="$1"
    
    # First ensure server is awake
    if ! wake_server; then
        echo "Cannot upload cookies file - server unavailable"
        return 1
    fi
    
    # Now proceed with upload
    echo "Uploading to remote server..."
    scp "$cookies_file" "$REMOTE_PATH"
    
    if [ $? -eq 0 ]; then
        echo "Upload successful"
        rm "$cookies_file"
        echo "Local file removed"
        return 0
    else
        echo "Upload failed"
        return 1
    fi
}

# Main file watching logic
fswatch -0 "$WATCH_DIR" | while read -d "" event
do
    filename=$(basename "$event")
    
    if [ "$filename" = "cookies.txt" ]; then
        echo "Found new cookies.txt file"
        sleep 1  # Brief pause to ensure file is fully written
        
        COOKIES_FILE="$WATCH_DIR/cookies.txt"
        handle_cookies_file "$COOKIES_FILE"
    fi
done