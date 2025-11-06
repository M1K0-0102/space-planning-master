# Git 自动安装脚本 (Windows PowerShell)
# 使用方法: 右键以管理员身份运行 PowerShell，然后执行此脚本
# 或在 PowerShell 中运行: .\install_git.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Git 自动安装脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否已安装 Git
Write-Host "正在检查 Git 是否已安装..." -ForegroundColor Yellow
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue

if ($gitInstalled) {
    $gitVersion = git --version
    Write-Host "✓ Git 已安装: $gitVersion" -ForegroundColor Green
    Write-Host ""
    
    $response = Read-Host "是否需要重新安装? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "跳过安装。" -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "准备安装 Git..." -ForegroundColor Yellow
Write-Host ""

# 方法选择
Write-Host "请选择安装方法:" -ForegroundColor Cyan
Write-Host "1. 使用 Chocolatey (推荐，自动安装)" -ForegroundColor White
Write-Host "2. 使用 Winget (Windows 包管理器)" -ForegroundColor White
Write-Host "3. 手动下载官方安装包" -ForegroundColor White
Write-Host ""

$choice = Read-Host "请输入选项 (1/2/3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "使用 Chocolatey 安装 Git..." -ForegroundColor Yellow
        
        # 检查是否已安装 Chocolatey
        $chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue
        
        if (-not $chocoInstalled) {
            Write-Host "Chocolatey 未安装，正在安装 Chocolatey..." -ForegroundColor Yellow
            
            # 安装 Chocolatey
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            try {
                Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
                Write-Host "✓ Chocolatey 安装成功" -ForegroundColor Green
            } catch {
                Write-Host "✗ Chocolatey 安装失败: $_" -ForegroundColor Red
                Write-Host "请尝试其他安装方法或访问: https://chocolatey.org/install" -ForegroundColor Yellow
                exit 1
            }
            
            # 刷新环境变量
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        }
        
        Write-Host "正在使用 Chocolatey 安装 Git..." -ForegroundColor Yellow
        try {
            choco install git -y
            Write-Host "✓ Git 安装成功！" -ForegroundColor Green
        } catch {
            Write-Host "✗ Git 安装失败: $_" -ForegroundColor Red
            exit 1
        }
    }
    
    "2" {
        Write-Host ""
        Write-Host "使用 Winget 安装 Git..." -ForegroundColor Yellow
        
        # 检查是否已安装 Winget
        $wingetInstalled = Get-Command winget -ErrorAction SilentlyContinue
        
        if (-not $wingetInstalled) {
            Write-Host "✗ Winget 未安装" -ForegroundColor Red
            Write-Host "Winget 是 Windows 11 和 Windows 10 (版本 1809+) 的内置工具" -ForegroundColor Yellow
            Write-Host "如果你的系统支持，请从 Microsoft Store 安装 '应用安装程序'" -ForegroundColor Yellow
            Write-Host "或访问: https://github.com/microsoft/winget-cli" -ForegroundColor Yellow
            exit 1
        }
        
        Write-Host "正在使用 Winget 安装 Git..." -ForegroundColor Yellow
        try {
            winget install --id Git.Git -e --source winget
            Write-Host "✓ Git 安装成功！" -ForegroundColor Green
        } catch {
            Write-Host "✗ Git 安装失败: $_" -ForegroundColor Red
            exit 1
        }
    }
    
    "3" {
        Write-Host ""
        Write-Host "准备下载 Git 官方安装包..." -ForegroundColor Yellow
        
        $downloadUrl = "https://github.com/git-for-windows/git/releases/latest/download/Git-2.43.0-64-bit.exe"
        $installerPath = "$env:TEMP\Git-Installer.exe"
        
        Write-Host "下载地址: $downloadUrl" -ForegroundColor Cyan
        Write-Host "保存位置: $installerPath" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "正在下载 Git 安装包..." -ForegroundColor Yellow
        
        try {
            # 使用进度条下载
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath
            $ProgressPreference = 'Continue'
            
            Write-Host "✓ 下载完成" -ForegroundColor Green
            Write-Host ""
            Write-Host "正在运行安装程序..." -ForegroundColor Yellow
            Write-Host "请在安装向导中完成安装（推荐使用默认选项）" -ForegroundColor Cyan
            
            # 运行安装程序
            Start-Process -FilePath $installerPath -Wait
            
            Write-Host "✓ 安装程序已运行" -ForegroundColor Green
            
            # 清理安装包
            Remove-Item $installerPath -ErrorAction SilentlyContinue
            
        } catch {
            Write-Host "✗ 下载或安装失败: $_" -ForegroundColor Red
            Write-Host "请手动访问 https://git-scm.com/download/win 下载安装" -ForegroundColor Yellow
            exit 1
        }
    }
    
    default {
        Write-Host "无效的选项，已取消安装。" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   配置 Git" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 刷新环境变量
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 验证安装
Write-Host "验证 Git 安装..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

try {
    $gitVersion = git --version 2>&1
    Write-Host "✓ $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git 命令不可用" -ForegroundColor Red
    Write-Host "可能需要重启 PowerShell 或计算机" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "配置 Git 用户信息..." -ForegroundColor Yellow
Write-Host ""

$userName = Read-Host "请输入你的用户名"
$userEmail = Read-Host "请输入你的邮箱"

if ($userName) {
    git config --global user.name "$userName"
    Write-Host "✓ 用户名已设置: $userName" -ForegroundColor Green
}

if ($userEmail) {
    git config --global user.email "$userEmail"
    Write-Host "✓ 邮箱已设置: $userEmail" -ForegroundColor Green
}

# 其他推荐配置
Write-Host ""
Write-Host "应用推荐配置..." -ForegroundColor Yellow

git config --global init.defaultBranch main
git config --global color.ui auto
git config --global core.quotepath false

Write-Host "✓ 默认分支设置为 main" -ForegroundColor Green
Write-Host "✓ 启用彩色输出" -ForegroundColor Green
Write-Host "✓ 修复中文文件名显示" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Git 配置信息:" -ForegroundColor Cyan
git config --list --global | Select-String "user.name|user.email|init.defaultBranch"

Write-Host ""
Write-Host "接下来你可以:" -ForegroundColor Cyan
Write-Host "  1. 重启 PowerShell 以确保环境变量生效" -ForegroundColor White
Write-Host "  2. 查看 GIT_SETUP_GUIDE.md 了解如何上传项目" -ForegroundColor White
Write-Host "  3. 查看 UPLOAD_CHECKLIST.md 使用上传检查清单" -ForegroundColor White
Write-Host ""

Write-Host "常用 Git 命令:" -ForegroundColor Cyan
Write-Host "  git --version          查看版本" -ForegroundColor White
Write-Host "  git config --list      查看配置" -ForegroundColor White
Write-Host "  git init              初始化仓库" -ForegroundColor White
Write-Host "  git status            查看状态" -ForegroundColor White
Write-Host ""

Read-Host "按回车键退出"

