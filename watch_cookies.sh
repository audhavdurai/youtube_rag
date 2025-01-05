#!/bin/bash

# Directory to watch
WATCH_DIR="$HOME/Downloads"
REMOTE_PATH="s_01jgsqrd7tefatat8ghx7wwhte@ssh.lightning.ai:./content/youtube_rag/backend/cookies.txt"

# Create fswatch command that watches for new files
fswatch -0 "$WATCH_DIR" | while read -d "" event
do
    # Get just the filename from the event
    filename=$(basename "$event")
    
    # Check if the file is cookies.txt
    if [ "$filename" = "cookies.txt" ]; then
        echo "Found new cookies.txt file"
        
        # Wait a brief moment to ensure file is completely written
        sleep 1
        
        # Full path to the cookies file
        COOKIES_FILE="$WATCH_DIR/cookies.txt"
        
        # Upload the file
        echo "Uploading to remote server..."
        scp "$COOKIES_FILE" "$REMOTE_PATH"
        
        # Check if scp was successful
        if [ $? -eq 0 ]; then
            echo "Upload successful"
            # Remove the local file
            rm "$COOKIES_FILE"
            echo "Local file removed"
        else
            echo "Upload failed"
        fi
    fi
done
