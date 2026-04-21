# 角色认知插件开发文档

## 架构概述

本插件采用模块化架构，将不同功能拆分为独立的模块，便于维护和扩展。

```
astrbot_plugin_self_recognition/
├── main.py                    # 插件主入口
├── modules/                   # 功能模块目录
│   ├── __init__.py           # 模块导出
│   ├── milvus_manager.py     # Milvus 数据库管理
│   ├── image_processor.py    # 图片处理和特征提取
│   ├── self_recognition.py   # 认识自己功能
│   └── character_recognition.py  # 认识角色功能
├── _conf_schema.json         # 插件配置模式
├── metadata.yaml             # 插件元数据
└── temp/                     # 临时文件目录（运行时创建）
```

## 模块说明

### 1. MilvusManager (`milvus_manager.py`)

负责 Milvus 向量数据库的连接、管理和操作。

**主要功能：**
- Milvus 数据库连接和认证
- 向量集合的创建和管理
- 向量的存储和检索
- 颜色特征缓存管理

**关键方法：**
```python
# 添加记忆
await milvus_manager.add_memory(
    text="特征描述",
    session_id="会话ID",
    character_type="self",  # 或 "character"
    character_name="角色名"  # 仅角色类型需要
)

# 搜索记忆
memories = await milvus_manager.search_memory(
    query_text="查询文本",
    session_id="会话ID",
    character_type="self",  # 可选过滤
    top_k=3
)
```

### 2. ImageProcessor (`image_processor.py`)

负责图片处理、视觉模型调用和特征提取。

**主要功能：**
- 图片数据解析和转换
- 调用视觉模型分析图片
- 提取角色特征（发色、眼白、眼瞳、种族特征等）
- 计算颜色相似度和特征相似度

**关键方法：**
```python
# 分析自己的形象
feature_desc = await image_processor.analyze_self_image(img_base64)

# 分析角色形象
feature_desc = await image_processor.analyze_character_image(img_base64, "角色名")

# 检测图片是否有人物
has_person = await image_processor.check_if_image_has_person(img_base64)

# 提取完整特征（发色、眼白、眼瞳、种族特征）
features = await image_processor.extract_color_features(feature_desc)

# 计算颜色相似度
similarity = image_processor.calculate_color_similarity("金色", "银色")

# 计算种族特征相似度
similarity = image_processor.calculate_feature_similarity("猫耳,尾巴", "猫耳")
```

### 3. SelfRecognitionHandler (`self_recognition.py`)

处理"认识自己"相关功能。

**主要功能：**
- 处理 `/认识自己` 指令的交互流程
- 检测图片中是否包含自己的形象
- 生成自我认知回复
- 获取对话上下文，让回复更自然

**关键方法：**
```python
# 处理认识自己指令
async for result in self_handler.handle_teach_self(event):
    yield result

# 处理图片检测自己
is_self, features = await self_handler.process_image_for_self(
    event, img_base64, session_id
)

# 生成自我认知回复
reply = await self_handler.generate_self_response(
    event, img_base64, self_features, user_text
)

# 获取最近对话上下文
context = await self_handler._get_recent_context(event, max_turns=5)
```

**回复生成特点（v2.0.2+）：**
- 模拟 AI 拥有视觉能力，以"我看到了..."的第一人称视角回应
- 自动获取最近 5 条对话历史，参照上下文生成回复
- **语言风格保持**：明确要求 LLM 保持与对话上下文一致的语言风格、格式和用语习惯
  - 自动学习并延续原有的说话方式（如括号描述动作、特殊表达习惯等）
  - 保持相同的句式结构、语气词和表达习惯
  - 让回复自然融入之前的对话风格中，提供更连贯的对话体验
- 优化 system_prompt 设计，让回复更加拟人化

### 4. CharacterRecognitionHandler (`character_recognition.py`)

处理"认识角色"相关功能。

**主要功能：**
- 处理 `/认识角色` 指令的交互流程
- 检测图片中是否包含已认识的角色
- 管理角色列表
- 生成角色认知回复
- 获取对话上下文，让回复更自然

**关键方法：**
```python
# 处理认识角色指令
async for result in character_handler.handle_teach_character(event):
    yield result

# 处理图片检测角色
is_character, name, features = await character_handler.process_image_for_character(
    event, img_base64, session_id
)

# 生成角色认知回复
reply = await character_handler.generate_character_response(
    event, img_base64, character_name, character_features, user_text
)

# 获取已认识角色列表
character_list = await character_handler.list_known_characters(event)

# 获取最近对话上下文
context = await character_handler._get_recent_context(event, max_turns=5)
```

**回复生成特点（v2.0.2+）：**
- 模拟 AI 拥有视觉能力，以"我看到了..."的第一人称视角回应
- 自动获取最近 5 条对话历史，参照上下文生成回复
- **语言风格保持**：明确要求 LLM 保持与对话上下文一致的语言风格、格式和用语习惯
  - 自动学习并延续原有的说话方式（如括号描述动作、特殊表达习惯等）
  - 保持相同的句式结构、语气词和表达习惯
  - 让回复自然融入之前的对话风格中，提供更连贯的对话体验
- 针对角色特点生成个性化的自然回复

## 数据结构

### Milvus 集合字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT64 | 主键（自增） |
| text | VARCHAR(2000) | 特征描述文本 |
| embedding | FLOAT_VECTOR | 向量表示 |
| session_id | VARCHAR(255) | 会话ID |
| persona_id | VARCHAR(255) | 人设ID |
| character_type | VARCHAR(50) | 角色类型（self/character） |
| character_name | VARCHAR(255) | 角色名称 |
| timestamp | INT64 | 时间戳 |

### 特征结构（v2.0.1+）

```python
{
    "hair_color": "银白色",       # 发色
    "eye_white_color": "白色",    # 眼白颜色
    "eye_pupil_color": "紫色",    # 眼瞳/虹膜颜色
    "racial_features": "猫耳,尾巴" # 种族特征（非人类特征）
}
```

**支持识别的种族特征：**
- 耳朵类：猫耳、狗耳、兔耳、狐耳、兽耳
- 角类：龙角、恶魔角、独角
- 尾巴类：尾巴、猫尾、狐尾、兔尾
- 翅膀类：翅膀、天使翼、恶魔翼
- 其他：兽人、人鱼、精灵、妖精

## 配置项

详见 `_conf_schema.json` 文件。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| vision_provider_id | string | - | 视觉模型提供商ID |
| embedding_api_key | string | - | 向量模型API Key |
| embedding_api_base | string | - | 向量模型API地址 |
| embedding_model | string | text-embedding-v4 | 向量模型名称 |
| embedding_dim | int | 1024 | 向量维度 |
| milvus_host | string | milvus-standalone | Milvus地址 |
| milvus_port | int | 19530 | Milvus端口 |
| milvus_db_name | string | default | 数据库名称 |
| milvus_token | string | - | Milvus Token |
| milvus_user | string | - | Milvus用户名 |
| milvus_password | string | - | Milvus密码 |
| collection_name | string | self_recognition_memory | 集合名称 |
| similarity_threshold | float | 0.65 | 向量相似度阈值 |
| hair_color_threshold | float | 0.8 | 发色相似度阈值 |
| eye_white_threshold | float | 0.8 | 眼白颜色相似度阈值 |
| eye_pupil_threshold | float | 0.8 | 眼瞳颜色相似度阈值 |
| racial_feature_threshold | float | 0.8 | 种族特征相似度阈值 |

## 识别流程

```
用户发送图片
    ↓
检测是否有人物/角色
    ↓ (是)
提取角色特征
    ↓
搜索"自己"的记忆
    ↓ (匹配)
颜色特征验证
    ↓ (通过)
生成自我认知回复
    ↓ (不匹配)
搜索"角色"的记忆
    ↓ (匹配)
颜色特征验证
    ↓ (通过)
生成角色认知回复
    ↓ (不匹配)
常规图片理解对话
```

## 扩展开发

### 添加新的角色类型

1. 在 `milvus_manager.py` 的 `_init_collection` 中添加新字段
2. 创建新的处理器模块，参考 `self_recognition.py`
3. 在 `main.py` 中注册新的指令和处理器

### 自定义特征提取

修改 `image_processor.py` 中的 `analyze_self_image` 或 `analyze_character_image` 方法的 prompt。

### 调整识别策略

修改 `self_recognition.py` 和 `character_recognition.py` 中的 `process_image_for_*` 方法。

## 注意事项

1. **向量维度**：确保配置的 `embedding_dim` 与向量模型的实际维度一致
2. **Milvus 认证**：根据部署方式选择正确的认证方式（Token 或用户名密码）
3. **视觉模型**：必须选择支持多模态的模型
4. **会话隔离**：记忆数据按 `session_id` 隔离，不同用户的数据互不干扰