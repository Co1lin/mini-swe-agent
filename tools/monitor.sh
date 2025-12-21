#!/bin/bash

# Configuration
LOG_FILE="system_usage.log"
INTERVAL=0.2

echo "Timestamp,CPU_%,MEM_%" > "$LOG_FILE"

echo "Monitoring started. Logging to $LOG_FILE... (Press Ctrl+C to stop)"

# Use a loop to capture usage every second
# 'top' is used in batch mode (-b) with 1 iteration (-n 1)
while true; do
    # Get current timestamp
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    # Extract CPU usage:
    # We subtract 'idle' percentage from 100 to get total usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | \
                sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | \
                awk '{print 100 - $1}')

    # Extract Memory usage:
    # Uses 'free' to calculate (Used / Total) * 100
    MEM_USAGE=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100}')

    # Log to file and stdout (optional)
    # The '>>' operator handles the append
    echo "$TIMESTAMP, $CPU_USAGE%, $MEM_USAGE%" >> "$LOG_FILE"

    # Ensure the write is flushed to disk immediately
    sync "$LOG_FILE"

    sleep "$INTERVAL"
done
