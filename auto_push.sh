#!/bin/bash
# auto_push.sh

# 1. 确保在项目根目录
cd "$(dirname "$0")"

# 2. 拉取最新远程代码，避免冲突
git pull --rebase origin master

# 3. 添加所有修改和新文件
git add -A

# 4. 提交，带时间戳信息
git commit -m "auto commit on $(date '+%Y-%m-%d %H:%M:%S')"

# 5. 推送到远程 master 分支
git push origin master