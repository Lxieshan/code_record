#!/bin/bash

log_dir="/path/to/log/directory"
days_to_keep=30  # 要保留的天数

# 获取当前日期之前的日期
delete_before=$(date -d "$days_to_keep days ago" +"%Y%m%d")

# 删除指定日期之前的日志文件
find "$log_dir" -type f -name "*.log" -exec bash -c '
    file_date=$(basename "$1" | grep -oE "[0-9]{8}")
    if [ "$file_date" -lt "'"$delete_before"'" ]; then
        echo "Deleting $1"
        rm "$1"
    fi
' _ {} \;
#!/usr/bin/env bash 

