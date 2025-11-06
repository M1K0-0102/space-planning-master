# 安全配置指南

## ⚠️ API Key 泄露紧急处理

如果你不小心将 API Key 提交到了 Git，请立即：

1. **撤销泄露的 API Key**（最重要！）
   - 访问 DeepSeek 控制台：https://platform.deepseek.com/api_keys
   - 删除泄露的 Key
   - 生成新的 Key

2. **从代码中移除 API Key**
3. **更新 Git 历史**（可选，但旧历史可能已被缓存）

## 🔒 安全的 API Key 配置方法

### 方法 1: 使用环境变量文件（推荐）

1. **复制示例文件**：
   ```bash
   copy env.example .env     # Windows
   # 或
   cp env.example .env       # Linux/Mac
   ```

2. **编辑 .env 文件，填入你的 API Key**：
   ```bash
   DEEPSEEK_API_KEY=sk-your-real-api-key-here
   ```

3. **确保 .env 在 .gitignore 中**（已配置）

4. **使用 python-dotenv 加载**（如果需要）：
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # 加载 .env 文件
   ```

### 方法 2: 使用系统环境变量

**Windows PowerShell**：
```powershell
$env:DEEPSEEK_API_KEY="sk-your-real-api-key-here"

# 永久设置（需要管理员权限）
[System.Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-your-real-api-key-here", "User")
```

**Linux/Mac**：
```bash
export DEEPSEEK_API_KEY="sk-your-real-api-key-here"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export DEEPSEEK_API_KEY="sk-your-real-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 方法 3: 代码中传入（不推荐用于生产）

```python
from src.pipeline.utils.suggestion_generator import SuggestionGenerator

# 仅用于测试，不要提交到 Git
generator = SuggestionGenerator(api_key="your-key-here")
```

## 🚫 永远不要做的事

❌ **不要**将 API Key 硬编码在代码中  
❌ **不要**将 API Key 提交到 Git  
❌ **不要**在公开场合分享 API Key  
❌ **不要**将 API Key 写在注释或文档中  
❌ **不要**将包含 API Key 的配置文件提交到 Git  

## ✅ 最佳实践

✅ 使用 `.env` 文件存储敏感信息  
✅ 确保 `.env` 在 `.gitignore` 中  
✅ 提供 `.env.example` 作为模板（不含真实 Key）  
✅ 定期轮换 API Key  
✅ 为不同环境使用不同的 Key  
✅ 使用环境变量或密钥管理服务  

## 🔍 检查代码是否包含敏感信息

在提交前，检查：

```bash
# 搜索可能的 API Key
git grep -i "api[_-]key\|secret\|password\|token"

# 查看将要提交的内容
git diff --cached
```

## 📚 相关资源

- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [GitHub 安全最佳实践](https://docs.github.com/cn/code-security)
- [.gitignore 生成器](https://www.toptal.com/developers/gitignore)

## 🆘 如果已经泄露

1. **立即撤销 API Key**
2. 生成新的 Key
3. 修改代码使用安全方式
4. 提交修复
5. 监控 API 使用情况，查看是否有异常调用

