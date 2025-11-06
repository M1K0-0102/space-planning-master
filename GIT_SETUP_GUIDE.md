# Git 安装和使用指南

本文档将指导你安装 Git 并将项目上传到 GitHub 或 Gitee。

## 📥 安装 Git

### Windows 系统

#### 方法 1: 官方安装包（推荐）

1. **下载 Git**
   - 访问 Git 官网: https://git-scm.com/download/win
   - 下载最新版本的 Git for Windows

2. **安装 Git**
   - 双击下载的安装包 (例如 `Git-2.43.0-64-bit.exe`)
   - 安装向导选项说明:
     - **Select Components**: 保持默认勾选即可
     - **Choosing the default editor**: 选择你喜欢的编辑器（推荐 Vim 或 VS Code）
     - **Adjusting your PATH environment**: 选择 "Git from the command line and also from 3rd-party software"（推荐）
     - **Choosing HTTPS transport backend**: 选择 "Use the OpenSSL library"
     - **Configuring the line ending conversions**: 选择 "Checkout Windows-style, commit Unix-style line endings"（推荐）
     - **Configuring the terminal emulator**: 选择 "Use MinTTY"
     - 其他选项保持默认

3. **验证安装**
   ```powershell
   # 打开 PowerShell 或 CMD
   git --version
   # 应该显示: git version 2.43.0.windows.1 (或更高版本)
   ```

#### 方法 2: 使用包管理器

**使用 Chocolatey**:
```powershell
# 以管理员身份运行 PowerShell
choco install git

# 验证安装
git --version
```

**使用 Scoop**:
```powershell
scoop install git
git --version
```

### Linux 系统

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install git

# 验证安装
git --version
```

#### CentOS/RHEL
```bash
sudo yum install git

# 或使用 dnf
sudo dnf install git

# 验证安装
git --version
```

### macOS 系统

#### 使用 Homebrew
```bash
brew install git

# 验证安装
git --version
```

## ⚙️ 配置 Git

首次使用 Git 需要设置用户信息：

```bash
# 设置用户名
git config --global user.name "你的名字"

# 设置邮箱
git config --global user.email "your.email@example.com"

# 查看配置
git config --list

# 设置默认分支名为 main
git config --global init.defaultBranch main
```

可选配置（提升使用体验）：

```bash
# 启用颜色输出
git config --global color.ui auto

# 设置默认编辑器
git config --global core.editor "code --wait"  # VS Code
# 或
git config --global core.editor "vim"  # Vim

# 配置别名（快捷命令）
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
```

## 🚀 创建仓库并上传项目

### 方案 1: 上传到 GitHub

#### 步骤 1: 创建 GitHub 仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息:
   - **Repository name**: `space-planning-master`
   - **Description**: 室内设计智能分析系统
   - **Public/Private**: 选择公开或私有
   - **不要勾选** "Initialize this repository with a README"
4. 点击 "Create repository"

#### 步骤 2: 初始化本地仓库

在项目目录下打开终端（PowerShell/CMD）：

```bash
# 进入项目目录
cd "D:\系统默认\文档\文件\项目\空间规划大师"

# 初始化 Git 仓库
git init

# 添加所有文件到暂存区
git add .

# 查看状态（确认文件已添加）
git status

# 提交到本地仓库
git commit -m "Initial commit: 空间规划大师项目初始版本"
```

#### 步骤 3: 关联远程仓库并推送

```bash
# 关联远程仓库（替换为你的 GitHub 用户名）
git remote add origin https://github.com/your-username/space-planning-master.git

# 推送到远程仓库
git branch -M main
git push -u origin main
```

如果推送时需要身份验证，有两种方式：

**方式 1: 使用 Personal Access Token (推荐)**

1. 在 GitHub 生成 Token:
   - 进入 Settings → Developer settings → Personal access tokens → Tokens (classic)
   - 点击 "Generate new token (classic)"
   - 勾选 `repo` 权限
   - 生成并复制 Token

2. 使用 Token 推送:
   ```bash
   # Windows: 使用凭据管理器
   git credential-manager-core configure
   git push -u origin main
   # 在弹出的窗口中输入用户名和 Token
   
   # 或直接在 URL 中使用 Token
   git remote set-url origin https://your-token@github.com/your-username/space-planning-master.git
   git push -u origin main
   ```

**方式 2: 使用 SSH**

```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your.email@example.com"

# 复制公钥内容
# Windows:
type %USERPROFILE%\.ssh\id_ed25519.pub
# Linux/Mac:
cat ~/.ssh/id_ed25519.pub

# 在 GitHub 添加 SSH Key:
# Settings → SSH and GPG keys → New SSH key
# 粘贴公钥内容并保存

# 修改远程仓库 URL 为 SSH 格式
git remote set-url origin git@github.com:your-username/space-planning-master.git
git push -u origin main
```

### 方案 2: 上传到 Gitee（推荐国内用户）

#### 步骤 1: 创建 Gitee 仓库

1. 登录 [Gitee](https://gitee.com)
2. 点击右上角 "+" → "新建仓库"
3. 填写仓库信息:
   - **仓库名称**: `space-planning-master`
   - **仓库介绍**: 室内设计智能分析系统
   - **是否开源**: 选择公开或私有
   - **不要勾选** "使用 Readme 文件初始化这个仓库"
4. 点击 "创建"

#### 步骤 2: 初始化本地仓库（同上）

```bash
cd "D:\系统默认\文档\文件\项目\空间规划大师"
git init
git add .
git commit -m "Initial commit: 空间规划大师项目初始版本"
```

#### 步骤 3: 关联 Gitee 并推送

```bash
# 关联远程仓库（替换为你的 Gitee 用户名）
git remote add origin https://gitee.com/your-username/space-planning-master.git

# 推送到远程仓库
git branch -M main
git push -u origin main
```

Gitee 身份验证：

```bash
# Gitee 支持 HTTPS 和 SSH 两种方式
# 推送时输入 Gitee 用户名和密码即可

# 或使用 SSH（同 GitHub 方法）
ssh-keygen -t ed25519 -C "your.email@example.com"
# 将公钥添加到 Gitee: 设置 → SSH 公钥
```

## 📋 上传前检查清单

在推送之前，确保：

- [ ] ✅ 已创建 `.gitignore` 文件
- [ ] ✅ 已创建 `README.md` 文档
- [ ] ✅ 已创建 `requirements.txt` 依赖列表
- [ ] ✅ 已排除 `__pycache__`、`output/`、大型模型文件
- [ ] ✅ 已提交有意义的 commit 信息
- [ ] ✅ 配置文件中不包含敏感信息（密码、密钥等）

验证排除文件：
```bash
# 查看将要提交的文件
git status

# 查看 .gitignore 是否生效
git check-ignore -v output/analysis_result_20250305_222021.json
# 应该显示该文件被忽略
```

## 🔄 日常 Git 操作

### 拉取最新代码

```bash
# 拉取远程更新
git pull origin main
```

### 提交新更改

```bash
# 查看修改的文件
git status

# 添加修改的文件
git add filename.py
# 或添加所有修改
git add .

# 提交更改
git commit -m "描述你的修改内容"

# 推送到远程
git push origin main
```

### 创建和切换分支

```bash
# 创建新分支
git branch feature-new-analyzer

# 切换分支
git checkout feature-new-analyzer

# 或一步创建并切换
git checkout -b feature-new-analyzer

# 推送新分支到远程
git push -u origin feature-new-analyzer

# 查看所有分支
git branch -a
```

### 查看历史记录

```bash
# 查看提交历史
git log

# 简洁格式查看
git log --oneline --graph --all

# 查看某个文件的修改历史
git log --follow filename.py
```

## 🤝 团队协作

### 邀请协作者

**GitHub**:
1. 进入仓库页面
2. Settings → Collaborators
3. 点击 "Add people"
4. 输入队友的 GitHub 用户名或邮箱

**Gitee**:
1. 进入仓库页面
2. 管理 → 仓库成员管理
3. 点击 "添加成员"
4. 输入队友的 Gitee 账号

### 队友克隆项目

```bash
# 克隆项目
git clone https://github.com/your-username/space-planning-master.git
# 或
git clone https://gitee.com/your-username/space-planning-master.git

# 进入项目目录
cd space-planning-master

# 按照 DEPLOYMENT.md 进行部署
```

### Pull Request 工作流

1. **创建功能分支**
   ```bash
   git checkout -b feature-xxx
   ```

2. **开发并提交**
   ```bash
   git add .
   git commit -m "Add feature xxx"
   git push origin feature-xxx
   ```

3. **在 GitHub/Gitee 上创建 Pull Request**
   - 进入仓库页面
   - 点击 "Pull requests" → "New pull request"
   - 选择源分支和目标分支
   - 填写 PR 描述并创建

4. **代码审查和合并**
   - 队友审查代码
   - 讨论并修改
   - 审查通过后合并到主分支

## 🐛 常见问题

### 1. Git 命令不被识别

**问题**: `'git' 不是内部或外部命令`

**解决**:
- 重新安装 Git，确保勾选添加到 PATH
- 或手动添加 Git 到系统环境变量:
  - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
  - 在 Path 中添加: `C:\Program Files\Git\cmd`

### 2. 推送被拒绝

**问题**: `error: failed to push some refs`

**解决**:
```bash
# 先拉取远程更新
git pull origin main --rebase

# 解决可能的冲突后，再次推送
git push origin main
```

### 3. 忘记添加 .gitignore

**问题**: 已经提交了不该提交的文件

**解决**:
```bash
# 创建 .gitignore 文件后
# 移除已跟踪的文件（但保留本地文件）
git rm -r --cached output/
git rm -r --cached __pycache__/
git rm --cached pretrained_models/*.pth

# 提交更改
git add .gitignore
git commit -m "Add .gitignore and remove tracked files"
git push origin main
```

### 4. 大文件推送失败

**问题**: `remote: error: File xxx.pth is 123.45 MB; this exceeds GitHub's file size limit`

**解决**:
```bash
# 方案 1: 使用 .gitignore 排除大文件（推荐）
# 在 .gitignore 中添加模型文件

# 方案 2: 使用 Git LFS（大文件存储）
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push origin main
```

### 5. 中文文件名显示乱码

**解决**:
```bash
# Windows 下配置
git config --global core.quotepath false
```

## 📚 学习资源

- **Git 官方文档**: https://git-scm.com/doc
- **GitHub 指南**: https://guides.github.com/
- **Gitee 帮助中心**: https://gitee.com/help
- **Pro Git 中文版**: https://git-scm.com/book/zh/v2

## 🎉 完成！

现在你已经成功安装 Git 并学会了如何上传项目到远程仓库。你的队友可以通过以下步骤开始使用项目：

1. 克隆仓库
2. 按照 `DEPLOYMENT.md` 安装依赖
3. 下载预训练模型
4. 开始开发

祝你和你的团队协作愉快！🚀

