# Git 上传检查清单

在将项目上传到 Git 之前，请按照此清单逐项检查。

## ✅ 上传前准备

### 1. 文件清理

- [ ] 删除或排除临时文件
  ```bash
  # 检查是否有临时文件
  dir /s *.tmp, *.bak, *.swp
  ```

- [ ] 排除编译文件和缓存
  - [ ] `__pycache__/` 目录
  - [ ] `*.pyc` 文件
  - [ ] `.egg-info/` 目录

- [ ] 排除输出文件
  - [ ] `output/` 目录下的所有分析结果
  - [ ] `*.log` 日志文件

- [ ] 排除大型模型文件
  - [ ] `pretrained_models/*.pth` (保留 `.gitkeep` 和 `places365_zh.txt`)
  - [ ] `pretrained_models/*.pt`

### 2. 必需文件检查

- [ ] ✅ `.gitignore` - 已创建
- [ ] ✅ `README.md` - 项目说明文档
- [ ] ✅ `requirements.txt` - 依赖列表
- [ ] ✅ `DEPLOYMENT.md` - 部署文档
- [ ] ✅ `GIT_SETUP_GUIDE.md` - Git 使用指南
- [ ] ✅ `LICENSE` - 许可证文件
- [ ] ✅ `setup.py` - 安装脚本

### 3. 代码检查

- [ ] 移除调试代码和 print 语句（非必需的）
- [ ] 移除硬编码的路径（使用相对路径或配置文件）
- [ ] 移除敏感信息
  - [ ] API 密钥
  - [ ] 密码
  - [ ] 个人信息
  - [ ] 数据库连接字符串

### 4. 文档检查

- [ ] README.md 包含：
  - [ ] 项目简介
  - [ ] 安装说明
  - [ ] 使用示例
  - [ ] 联系方式（更新为你的信息）

- [ ] DEPLOYMENT.md 包含：
  - [ ] 详细的部署步骤
  - [ ] 模型下载链接
  - [ ] 常见问题解决方案

## 🚀 上传步骤

### 步骤 1: 验证 .gitignore

```bash
# 查看将要提交的文件
git status

# 确认大文件被排除
git check-ignore -v pretrained_models/resnet50_places365.pth
git check-ignore -v output/analysis_result_20250305_222021.json
```

### 步骤 2: 初始化仓库

```bash
# 初始化 Git
git init

# 添加所有文件
git add .

# 查看将要提交的文件列表
git status

# 确认没有不该提交的文件
```

### 步骤 3: 首次提交

```bash
# 提交到本地仓库
git commit -m "Initial commit: 空间规划大师项目初始版本

- 实现五大核心分析功能（场景、家具、光照、色彩、风格）
- 添加完整的项目文档
- 包含测试用例和示例
- 配置文件和安装脚本"
```

### 步骤 4: 关联远程仓库

选择 GitHub 或 Gitee：

**GitHub:**
```bash
git remote add origin https://github.com/your-username/space-planning-master.git
git branch -M main
git push -u origin main
```

**Gitee:**
```bash
git remote add origin https://gitee.com/your-username/space-planning-master.git
git branch -M main
git push -u origin main
```

## 📋 上传后检查

### 1. 验证远程仓库

- [ ] 访问仓库页面，确认文件已上传
- [ ] 检查 README.md 是否正确显示
- [ ] 确认 `.gitignore` 生效（大文件未上传）
- [ ] 测试克隆仓库：
  ```bash
  cd ../
  git clone <your-repo-url> test-clone
  cd test-clone
  ```

### 2. 更新仓库信息

在 GitHub/Gitee 上：

- [ ] 添加项目描述
- [ ] 添加项目标签/主题
  - `python`
  - `deep-learning`
  - `pytorch`
  - `interior-design`
  - `computer-vision`
- [ ] 设置仓库主页（如果有）
- [ ] 启用 Issues（方便反馈问题）

### 3. 邀请协作者

- [ ] 在仓库设置中添加团队成员
- [ ] 发送仓库链接给队友
- [ ] 提供 `DEPLOYMENT.md` 文档

## 📤 提供给队友的信息

将以下信息发送给你的队友：

```
🎉 项目已上传到 Git！

📦 仓库地址:
GitHub: https://github.com/your-username/space-planning-master
或
Gitee: https://gitee.com/your-username/space-planning-master

📚 快速开始:
1. 克隆仓库: git clone <仓库地址>
2. 安装 Git: 参考项目中的 GIT_SETUP_GUIDE.md
3. 部署项目: 参考项目中的 DEPLOYMENT.md

📥 模型文件:
由于模型文件较大，没有上传到 Git
请从以下位置获取:
- 网盘链接: [提供网盘链接]
- 提取码: [提取码]
下载后解压到 pretrained_models/ 目录

💡 使用说明:
- README.md - 项目概览和基本使用
- DEPLOYMENT.md - 详细的部署指南
- docs/ 目录 - 技术文档和 API 文档

❓ 遇到问题:
- 查看 DEPLOYMENT.md 的常见问题部分
- 在仓库中提 Issue
- 联系我: [你的联系方式]
```

## 🔍 文件大小检查

上传前确认文件大小：

```bash
# 检查大文件（超过 10MB）
# Windows PowerShell
Get-ChildItem -Recurse | Where-Object {$_.Length -gt 10MB} | Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB, 2)}}

# Linux/Mac
find . -type f -size +10M -exec ls -lh {} \;
```

**注意**: GitHub 单文件限制 100MB，Gitee 单文件限制 50MB

如果有大文件：
- 确保在 `.gitignore` 中排除
- 或使用 Git LFS 管理

## ✨ 最终检查

- [ ] 仓库可以成功克隆
- [ ] 按照 DEPLOYMENT.md 可以成功部署
- [ ] README.md 显示正常
- [ ] 所有链接有效
- [ ] 联系信息已更新
- [ ] 队友已收到通知

## 🎊 完成！

恭喜！你的项目已成功上传到 Git。

现在你和你的团队可以：
- ✅ 协同开发
- ✅ 版本控制
- ✅ 代码审查
- ✅ 问题追踪

继续加油！🚀

