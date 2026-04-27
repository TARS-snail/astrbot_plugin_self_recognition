# 角色认知插件

[![Version](https://img.shields.io/badge/version-2.1.2-blue.svg)](https://github.com/TARS-snail/astrbot_plugin_self_recognition)
[![AstrBot](https://img.shields.io/badge/AstrBot-3.5+-green.svg)](https://github.com/AstrBotDevs/AstrBot-desktop)

> 让 AI 认识自己和其他角色的形象，自动识别图片中的角色并以自然方式回应

## ✨ 功能特点

- 🪞 **认识自己**：通过 `/认识自己` 指令，让 AI 记住自己的形象，后续能自动识别图片中是否有自己
- 👥 **认识角色**：通过 `/认识角色` 指令，让 AI 记住其他角色的形象（如初音未来、蕾姆等）
- 🎯 **四重特征验证**：采用向量相似度 + 发色/眼白/眼瞳/种族特征四重验证，提高识别准确性
- 🐱 **种族特征识别**：支持识别猫耳、龙角、尾巴、翅膀等非人类特征
- 🤖 **自动识别**：用户发送图片时，自动检测是否包含自己或已认识的角色
- 💬 **自然对话**：识别成功后，结合图片内容和角色特征进行自然对话

## 📦 安装

### 前置要求

1. **Milvus 向量数据库**
   - 本地部署：[Milvus 安装文档](https://milvus.io/docs/install_standalone-docker.md)
   - 云服务：[Zilliz Cloud](https://zilliz.com/)

2. **视觉模型提供商**
   - 支持多模态的 LLM 提供商（如 qwen-vl-plus、gpt-4-vision 等）

3. **向量模型 API**
   - 阿里云 DashScope API Key（推荐）
   - 或其他 OpenAI 兼容的 Embedding API

### 安装插件

将插件目录放入 AstrBot 的 `addons/plugins/` 目录下，重启 AstrBot 即可。

## ⚙️ 配置

在 AstrBot WebUI 的插件配置页面进行配置：

| 配置项 | 必填 | 说明 |
|--------|------|------|
| vision_provider_id | ✅ | 视觉模型提供商ID，选择支持多模态的模型 |
| embedding_api_key | ✅ | 向量模型 API Key |
| embedding_api_base | ❌ | 向量模型 API 地址，留空使用默认阿里云地址 |
| embedding_model | ❌ | 向量模型名称，默认 text-embedding-v4 |
| embedding_dim | ❌ | 向量维度，默认 1024 |
| milvus_host | ✅ | Milvus 服务地址 |
| milvus_port | ✅ | Milvus 端口 |
| milvus_db_name | ❌ | 数据库名称，默认 default |
| milvus_token | ❌ | Zilliz Cloud 的 API Key |
| milvus_user | ❌ | Milvus 用户名（开启认证时需要） |
| milvus_password | ❌ | Milvus 密码（开启认证时需要） |
| collection_name | ❌ | 集合名称，默认 self_recognition_memory |
| similarity_threshold | ❌ | 向量相似度阈值，默认 0.65 |
| hair_color_threshold | ❌ | 发色相似度阈值，默认 0.8 |
| eye_white_threshold | ❌ | 眼白颜色相似度阈值，默认 0.8 |
| eye_pupil_threshold | ❌ | 眼瞳颜色相似度阈值，默认 0.8 |
| racial_feature_threshold | ❌ | 种族特征相似度阈值，默认 0.8 |

## 🚀 使用方法

### 指令列表

| 指令 | 说明 |
|------|------|
| `/认识自己` | 教 AI 认识自己的形象 |
| `/认识角色` | 教 AI 认识其他角色的形象 |
| `/角色列表` | 查看已认识的角色列表 |
| `/角色认知设置` | 查看当前插件配置 |

### 使用流程

1. **认识自己**
   ```
   用户: /认识自己
   AI: 📸 好的，请发送一张我的照片～
   用户: [发送图片]
   AI: ✅ 认识啦！我的形象特征：银白色长发，紫色瞳孔，双马尾...
   ```

2. **认识角色**
   ```
   用户: /认识角色
   AI: 📸 好的，请先告诉我这个角色的名字～
   用户: 初音未来
   AI: 📝 角色名：初音未来，现在请发送一张初音未来的形象图片～
   用户: [发送图片]
   AI: ✅ 已记住初音未来！角色特征：青绿色双马尾，青绿色瞳孔...
   ```

3. **自动识别**
   ```
   用户: [发送一张 AI 形象的图片]
   AI: 咦，这是我诶！这张照片里的我看起来很开心呢～
   
   用户: [发送一张初音未来的图片]
   AI: 这是初音未来吧！我认识ta的！青绿色的双马尾很有辨识度呢～
   ```

## 🔧 技术架构

```
用户发送图片
    ↓
检测是否有人物/角色
    ↓ (是)
提取角色特征（发色、瞳色、发型等）
    ↓
搜索"自己"的记忆
    ↓ (匹配成功)
颜色特征验证 → 通过 → 生成自我认知回复
    ↓ (不匹配)
搜索"角色"的记忆
    ↓ (匹配成功)
颜色特征验证 → 通过 → 生成角色认知回复
    ↓ (不匹配)
常规图片理解对话
```

## 📁 项目结构

```
astrbot_plugin_self_recognition/
├── main.py                    # 插件主入口
├── modules/                   # 功能模块
│   ├── milvus_manager.py     # Milvus 数据库管理
│   ├── image_processor.py    # 图片处理和特征提取
│   ├── self_recognition.py   # 认识自己功能
│   └── character_recognition.py  # 认识角色功能
├── _conf_schema.json         # 插件配置模式
├── metadata.yaml             # 插件元数据
├── CHANGELOG.md              # 更新日志
├── DEVELOPMENT.md            # 开发文档
└── README.md                 # 说明文档
```

## 🔄 更新日志

### v2.1.1
- 🐛 **修复 `/角色列表` 报错**：原代码使用空字符串查询向量搜索导致 Embedding API 报错，改用 Milvus 直接查询（不需要向量搜索）
- 🛡️ **防御性检查**：在 `get_embedding` 方法入口增加空文本检查，防止空字符串被发送到 Embedding API

### v2.1.0
- 💾 **对话历史存入**：将图片识别结果和AI回复存入对话历史，解决AI不知道自己看过图片导致对话跳跃的问题
  - 识别到自己时：存入用户消息（包含图片描述和自身特征）和AI回复
  - 识别到角色时：存入用户消息（包含图片描述和角色特征）和AI回复
  - 常规识图时：存入用户消息（包含图片内容描述）和AI回复
  - 后续对话中AI可以回顾之前看过什么图片、识别了什么角色，保持对话连贯性

### v2.0.2
- 🗣️ **语言风格保持**：在所有回复生成时，明确要求 LLM 保持与对话上下文一致的语言风格、格式和用语习惯
  - 自动学习并延续原有的说话方式（如括号描述动作、特殊表达习惯等）
  - 保持相同的句式结构、语气词和表达习惯
  - 让回复自然融入之前的对话风格中，提供更连贯的对话体验

### v2.0.1
- 👁️ **图片消息优化**：模拟 AI 拥有视觉能力，以"我看到了..."的第一人称视角回应
- 💬 **上下文回顾**：自动获取最近 5 条对话历史，让回复更加自然连贯

### v2.0.0
- ✨ 新增认识角色功能
- 🔨 模块化重构，便于维护
- 🐛 修复数据库连接配置错误
- 📝 新增开发文档

详见 [CHANGELOG.md](./CHANGELOG.md)

## 📖 开发文档

详见 [DEVELOPMENT.md](./DEVELOPMENT.md)

## ❓ 常见问题

**Q: Milvus 连接失败怎么办？**

A: 请检查以下内容：
- Milvus 服务是否正常运行
- 地址和端口是否正确
- 如果开启了认证，是否配置了正确的用户名密码或 Token

**Q: 识别不准确怎么办？**

A: 可以尝试：
- 提高相似度阈值（如 0.7）
- 提供更多不同角度的形象图片
- 确保视觉模型能正确识别二次元角色

**Q: 支持哪些向量模型？**

A: 支持所有 OpenAI 兼容的 Embedding API，推荐使用阿里云 DashScope 的 text-embedding-v4。

## 📄 许可证

MIT License

## 🙏 致谢

- [AstrBot](https://github.com/AstrBotDevs/AstrBot-desktop) - 强大的机器人框架
- [Milvus](https://milvus.io/) - 高性能向量数据库