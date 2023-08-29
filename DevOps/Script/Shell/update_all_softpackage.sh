#!/bin/bash

# 服务器IP地址列表文件
server_list="server_ips.txt"

# 循环遍历服务器列表并执行更新命令
while IFS= read -r server_ip; do
    echo "Updating packages on $server_ip"
    ssh user@$server_ip "sudo apt update && sudo apt upgrade -y"
    echo "Update on $server_ip complete"
    echo
done < "$server_list"
#!/usr/bin/env bash 

