# Save as watchdog.sh
#!/bin/bash

LOG_FILE="/home/nemes1s/Desktop/cctv-detection/prototype-1/logs/watchdog.log"
PID_FILE="/tmp/cctv_detection.pid"
SERVICE_NAME="cctv-detection.service"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

check_service() {
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        return 0
    else
        return 1
    fi
}

check_memory_usage() {
    MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
    MEMORY_THRESHOLD=90.0
    
    if (( $(echo "$MEMORY_USAGE > $MEMORY_THRESHOLD" | bc -l) )); then
        log_message "WARNING: High memory usage: ${MEMORY_USAGE}%"
        return 1
    fi
    return 0
}

check_disk_space() {
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    DISK_THRESHOLD=95
    
    if [ "$DISK_USAGE" -gt "$DISK_THRESHOLD" ]; then
        log_message "WARNING: High disk usage: ${DISK_USAGE}%"
        # Clean up old clips
        find /home/nemes1s/Desktop/cctv-detection/output_clips -name "*.mp4" -mtime +7 -delete
        return 1
    fi
    return 0
}

check_output_directory() {
    OUTPUT_DIR="/home/nemes1s/Desktop/cctv-detection/output_clips"
    CLIP_COUNT=$(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    MAX_CLIPS=1000
    
    if [ "$CLIP_COUNT" -gt "$MAX_CLIPS" ]; then
        log_message "Too many clips ($CLIP_COUNT), cleaning up oldest..."
        find "$OUTPUT_DIR" -name "*.mp4" -type f -printf '%T@ %p\n' | sort -n | head -n 500 | cut -d' ' -f2- | xargs rm -f
    fi
}

main() {
    log_message "Watchdog check started"
    
    # Check if service is running
    if ! check_service; then
        log_message "ERROR: Service is not running, attempting restart..."
        sudo systemctl restart "$SERVICE_NAME"
        sleep 10
        
        if check_service; then
            log_message "Service restarted successfully"
        else
            log_message "CRITICAL: Service restart failed"
        fi
    fi
    
    # Check system resources
    check_memory_usage
    check_disk_space
    check_output_directory
    
    log_message "Watchdog check completed"
}

# Run checks
main
