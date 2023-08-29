#!/bin/bash

log_file="/path/to/status.log"

# 获取当前日期和时间
timestamp=$(date +"%Y-%m-%d %H:%M:%S")

# 获取系统资源使用情况
cpu_usage=$(top -b -n 1 | grep "Cpu(s)" | awk '{print $2 + $4}')
memory_usage=$(free -m | awk '/Mem:/ {print $3}')
disk_usage=$(df -h / | awk '/\// {print $5}')

# 检查特定服务状态
check_service() {
    service_name=$1
    service_status=$(systemctl is-active "$service_name")
    echo "Service $service_name is $service_status"
}

# 将信息写入日志文件
echo "$timestamp - CPU Usage: $cpu_usage%, Memory Usage: $memory_usage MB, Disk Usage: $disk_usage" >> "$log_file"
check_service "nginx" >> "$log_file"
check_service "mysql" >> "$log_file"
# 添加更多的服务检查

echo "---------------------------------------" >> "$log_file"
#!/usr/bin/env bash 

