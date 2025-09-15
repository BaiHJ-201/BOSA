#!/bin/bash
git config --global user.name "nocoding-maker"

# 进入项目目录
cd /root/TRIBE || exit

# 确保远程仓库已设置
git remote -v || git remote add origin git@github.com:nocoding-maker/Real-TTA-in-edge.git

# 拉取最新远程更新，避免冲突
git pull origin master --rebase

# 添加更新（忽略 datasets/）
git add .

# 提交，自动带时间戳
git commit -m "Auto update: $(date '+%Y-%m-%d %H:%M:%S')"

# 推送到远程
git push -u origin master