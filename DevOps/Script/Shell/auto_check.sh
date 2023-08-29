#!/bin/bash

website_url="https://example.com"
notification_email="admin@example.com"
check_interval=300  # 检测间隔（以秒为单位）

while true; do
    response=$(curl -Is "$website_url" | head -n 1)
    
    if [[ ! $response =~ "200 OK" ]]; then
        echo "Website is not accessible. Sending notification..."
        echo "Website is not accessible. Response: $response" | mail -s "Website Down" "$notification_email"
    else
        echo "Website is accessible."
    fi
    
    sleep "$check_interval"
done
#!/usr/bin/env bash 

