# Save as monitor.sh
#!/bin/bash
while true; do
    if ! systemctl is-active --quiet cctv-detection.service; then
        echo "$(date): Service is down, restarting..."
        sudo systemctl restart cctv-detection.service
    fi
    sleep 60  # Check every minute
done
