"""
Milvus 数据库管理模块
负责 Milvus 连接、向量存储和检索
"""
import aiohttp
from typing import List, Dict, Optional
from pathlib import Path

from astrbot.api import logger

# Milvus 相关
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)


class MilvusManager:
    """Milvus 数据库管理器"""
    
    def __init__(self, config: dict):
        """
        初始化 Milvus 管理器
        
        Args:
            config: 插件配置字典
        """
        # Milvus 配置 - 修复: 使用正确的配置键名 milvus_db_name
        self.milvus_host = config.get("milvus_host", "localhost")
        self.milvus_port = config.get("milvus_port", 19530)
        self.milvus_db_name = config.get("milvus_db_name", "default")  # 修复: 原来错误地使用了 milvus_db_id
        self.milvus_token = config.get("milvus_token", "")
        self.milvus_user = config.get("milvus_user", "")
        self.milvus_password = config.get("milvus_password", "")
        self.collection_name = config.get("collection_name", "self_recognition_memory")
        
        # 向量模型配置
        self.custom_embedding_api_key = config.get("embedding_api_key", "")
        self.custom_embedding_api_base = config.get("embedding_api_base", "")
        self.custom_embedding_model = config.get("embedding_model", "text-embedding-v4")
        self.custom_embedding_dim = config.get("embedding_dim", 1024)
        
        # 判断是否使用自定义向量模型
        self.use_custom_embedding = bool(self.custom_embedding_api_key)
        
        # 颜色特征缓存 (session_id -> color_features)
        self.color_features_cache = {}
        
        # Milvus 连接状态
        self._milvus_connected = False
        self._milvus_error = None
        self.collection = None
        
        # 尝试初始化连接
        self._try_connect_milvus()
    
    def _try_connect_milvus(self):
        """尝试连接 Milvus，失败不阻止插件加载"""
        try:
            self._connect_milvus()
            self._init_collection()
            self._milvus_connected = True
        except Exception as e:
            self._milvus_error = str(e)
            logger.warning(f"[角色认知] Milvus 连接初始化失败，插件将加载但功能受限: {e}")
            logger.warning("[角色认知] 请检查 Milvus 配置和认证信息，修复后重启 AstrBot")
    
    def _ensure_milvus_connected(self) -> bool:
        """确保 Milvus 已连接，用于实际使用前检查"""
        if self._milvus_connected:
            return True
        
        # 尝试重新连接
        try:
            self._connect_milvus()
            self._init_collection()
            self._milvus_connected = True
            return True
        except Exception as e:
            logger.error(f"[角色认知] Milvus 连接失败: {e}")
            return False
    
    def _connect_milvus(self):
        """连接到 Milvus 数据库"""
        # 构建连接参数，支持多种认证方式
        connect_params = {
            "alias": "default",
            "host": self.milvus_host,
            "port": self.milvus_port,
            "db_name": self.milvus_db_name
        }
        
        # 优先使用 token（适用于 Zilliz Cloud 等）
        if self.milvus_token:
            connect_params["token"] = self.milvus_token
            logger.info("[角色认知] 使用 Token 认证连接 Milvus")
        # 否则使用用户名密码认证
        elif self.milvus_user and self.milvus_password:
            connect_params["user"] = self.milvus_user
            connect_params["password"] = self.milvus_password
            logger.info(f"[角色认知] 使用用户名密码认证连接 Milvus: {self.milvus_user}")
        # 无认证配置时，尝试使用 Milvus 默认认证（root/Milvus）
        else:
            # Milvus 开启认证后的默认用户名密码
            connect_params["user"] = "root"
            connect_params["password"] = "Milvus"
            logger.info("[角色认知] 未配置认证信息，尝试使用默认认证 (root/Milvus)")
        
        connections.connect(**connect_params)
        logger.info(f"[角色认知] 已连接 Milvus: {self.milvus_host}:{self.milvus_port}, 数据库: {self.milvus_db_name}")
    
    def _get_embedding_dim_from_config(self) -> int:
        """获取配置的向量维度"""
        if self.custom_embedding_dim:
            return self.custom_embedding_dim
        return 1024
    
    def _init_collection(self):
        """初始化 Milvus 集合"""
        embedding_dim = self._get_embedding_dim_from_config()
        
        if utility.has_collection(self.collection_name, using="default"):
            self.collection = Collection(self.collection_name, using="default")
            # 验证现有集合的维度是否匹配配置
            schema = self.collection.schema
            for field in schema.fields:
                if field.name == "embedding":
                    existing_dim = field.params.get("dim")
                    if existing_dim != embedding_dim:
                        logger.warning(f"[角色认知] 集合维度不匹配！配置维度: {embedding_dim}, 集合维度: {existing_dim}")
                        logger.warning("[角色认知] 这可能导致向量搜索失败，请确保配置的 embedding_dim 与集合维度一致")
                    break
            logger.info(f"[角色认知] 集合 {self.collection_name} 已存在，维度: {embedding_dim}")
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="persona_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="character_type", dtype=DataType.VARCHAR, max_length=50),  # 新增: 区分是自己还是角色
            FieldSchema(name="character_name", dtype=DataType.VARCHAR, max_length=255),  # 新增: 角色名称
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, description="角色认知记忆集合")
        self.collection = Collection(self.collection_name, schema, using="default")
        
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"[角色认知] 集合 {self.collection_name} 创建成功，维度: {embedding_dim}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        if not text or not text.strip():
            raise ValueError("嵌入查询文本不能为空")
        if not self.use_custom_embedding:
            raise ValueError("请配置自定义向量模型 API Key")
        return await self._get_embedding_custom(text)
    
    async def _get_embedding_custom(self, text: str) -> List[float]:
        """使用自定义向量模型获取向量"""
        if not self.custom_embedding_api_key:
            raise ValueError("自定义向量模型未配置 API Key")
        
        # 支持多种 API 格式
        if self.custom_embedding_api_base:
            base_url = self.custom_embedding_api_base.rstrip('/')
        else:
            # 阿里云 DashScope OpenAI 兼容模式
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 检查 base_url 是否已包含 /v1 或其他路径
        if base_url.endswith('/embeddings'):
            url = base_url
        elif base_url.endswith('/v1'):
            url = f"{base_url}/embeddings"
        else:
            url = f"{base_url}/v1/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.custom_embedding_api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        payload = {
            "model": self.custom_embedding_model,
            "input": text
        }
        # 只有 OpenAI 兼容 API 支持 dimensions 参数
        if self.custom_embedding_dim and "openai" not in url.lower():
            pass
        elif self.custom_embedding_dim:
            payload["dimensions"] = self.custom_embedding_dim
        
        logger.debug(f"[角色认知] Embedding API 请求: {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"[角色认知] Embedding API 错误响应: {error_text}")
                        raise Exception(f"HTTP {resp.status}: {error_text}")
                    result = await resp.json()
                    
                    if "data" not in result or len(result["data"]) == 0:
                        raise Exception(f"API 返回格式异常: {result}")
                    
                    embedding = result["data"][0]["embedding"]
                    logger.info(f"[角色认知] Embedding 成功，维度: {len(embedding)}")
                    return embedding
        except aiohttp.ClientError as e:
            logger.error(f"[角色认知] 网络请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"[角色认知] 自定义 Embedding 调用失败: {e}")
            raise
    
    async def add_memory(
        self, 
        text: str, 
        session_id: str, 
        character_type: str = "self",
        character_name: str = "",
        persona_id: str = ""
    ) -> bool:
        """
        添加记忆到 Milvus
        
        Args:
            text: 特征描述文本
            session_id: 会话ID
            character_type: 角色类型 ("self" 或 "character")
            character_name: 角色名称
            persona_id: 人设ID
            
        Returns:
            是否添加成功
        """
        try:
            embedding = await self.get_embedding(text)
            import time
            data = [
                [text],
                [embedding],
                [session_id],
                [persona_id],
                [character_type],
                [character_name],
                [int(time.time())]
            ]
            self.collection.insert(data)
            self.collection.flush()
            logger.info(f"[角色认知] 记忆添加成功: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"[角色认知] 添加记忆失败: {e}")
            return False
    
    async def search_memory(
        self, 
        query_text: str, 
        session_id: str, 
        character_type: str = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        搜索记忆
        
        Args:
            query_text: 查询文本
            session_id: 会话ID
            character_type: 角色类型过滤 (可选)
            top_k: 返回结果数量
            
        Returns:
            匹配的记忆列表
        """
        try:
            query_vec = await self.get_embedding(query_text)
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            self.collection.load()
            
            # 构建过滤表达式
            expr_parts = [f'session_id == "{session_id}"']
            if character_type:
                expr_parts.append(f'character_type == "{character_type}"')
            expr = " && ".join(expr_parts)
            
            results = self.collection.search(
                data=[query_vec],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "session_id", "persona_id", "character_type", "character_name", "timestamp"]
            )
            
            memories = []
            if results:
                for hits in results:
                    for hit in hits:
                        memories.append({
                            "text": hit.entity.get("text"),
                            "score": hit.score,
                            "session_id": hit.entity.get("session_id"),
                            "persona_id": hit.entity.get("persona_id"),
                            "character_type": hit.entity.get("character_type"),
                            "character_name": hit.entity.get("character_name"),
                            "timestamp": hit.entity.get("timestamp")
                        })
            return memories
        except Exception as e:
            logger.error(f"[角色认知] 检索失败: {e}")
            return []
    
    async def query_memories(
        self,
        session_id: str,
        character_type: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        直接查询记忆（不需要向量搜索，适用于列表查询等场景）
        
        Args:
            session_id: 会话ID
            character_type: 角色类型过滤 (可选)
            limit: 返回结果数量
            
        Returns:
            匹配的记忆列表
        """
        try:
            if not self._ensure_milvus_connected():
                logger.error("[角色认知] Milvus 未连接，无法查询记忆")
                return []
            
            self.collection.load()
            
            # 构建过滤表达式
            expr_parts = [f'session_id == "{session_id}"']
            if character_type:
                expr_parts.append(f'character_type == "{character_type}"')
            expr = " && ".join(expr_parts)
            
            results = self.collection.query(
                expr=expr,
                output_fields=["text", "session_id", "persona_id", "character_type", "character_name", "timestamp"],
                limit=limit
            )
            
            memories = []
            if results:
                for result in results:
                    memories.append({
                        "text": result.get("text"),
                        "session_id": result.get("session_id"),
                        "persona_id": result.get("persona_id"),
                        "character_type": result.get("character_type"),
                        "character_name": result.get("character_name"),
                        "timestamp": result.get("timestamp")
                    })
            return memories
        except Exception as e:
            logger.error(f"[角色认知] 查询记忆失败: {e}")
            return []
    
    def update_color_cache(self, session_id: str, color_features: Dict[str, str]):
        """更新颜色特征缓存"""
        self.color_features_cache[session_id] = color_features
        logger.info(f"[角色认知] 颜色特征缓存更新: {session_id} -> {color_features}")
    
    def get_color_cache(self, session_id: str) -> Dict[str, str]:
        """获取颜色特征缓存"""
        return self.color_features_cache.get(session_id, {})
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._milvus_connected
    
    def get_error(self) -> Optional[str]:
        """获取连接错误信息"""
        return self._milvus_error
    
    def disconnect(self):
        """断开 Milvus 连接"""
        try:
            connections.disconnect("default")
            logger.info("[角色认知] Milvus 连接已断开")
        except:
            pass