"""
角色认知插件 - 主入口
版本: 2.1.1
作者: TARS_snail

让 AI 认识你希望ta记住的形象，自动知道用户发送的图片中是否有这些角色，并以第一人称回应。
"""
import json
from pathlib import Path
from typing import List

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Image, Plain
from astrbot.core.agent.message import AssistantMessageSegment, UserMessageSegment, TextPart

from .modules.milvus_manager import MilvusManager
from .modules.image_processor import ImageProcessor
from .modules.self_recognition import SelfRecognitionHandler
from .modules.character_recognition import CharacterRecognitionHandler


@register("astrbot_plugin_self_recognition", "TARS_snail", "角色认知插件——让AI认出图片中的自己和其他角色", "2.1.1")
class SelfRecognitionPlugin(Star):
    """角色认知插件主类"""
    
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        
        # 配置参数
        self.vision_provider_id = self.config.get("vision_provider_id", "")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.65)
        
        # 各项特征阈值配置
        self.hair_color_threshold = self.config.get("hair_color_threshold", 0.8)
        self.eye_white_threshold = self.config.get("eye_white_threshold", 0.8)
        self.eye_pupil_threshold = self.config.get("eye_pupil_threshold", 0.8)
        self.racial_feature_threshold = self.config.get("racial_feature_threshold", 0.8)
        
        # 初始化模块
        self._init_modules()
        
        logger.info("[角色认知] 插件初始化完成 v2.1.1")
        logger.info(f"[角色认知] 视觉模型提供商: {self.vision_provider_id}")
        logger.info(f"[角色认知] 特征阈值 - 发色: {self.hair_color_threshold}, 眼白: {self.eye_white_threshold}, 眼瞳: {self.eye_pupil_threshold}, 种族: {self.racial_feature_threshold}")
    
    def _init_modules(self):
        """初始化各功能模块"""
        # Milvus 管理器
        self.milvus_manager = MilvusManager(self.config)
        
        # 图片处理器
        self.image_processor = ImageProcessor(
            context=self.context,
            vision_provider_id=self.vision_provider_id,
            similarity_threshold=self.similarity_threshold,
            hair_color_threshold=self.hair_color_threshold,
            eye_white_threshold=self.eye_white_threshold,
            eye_pupil_threshold=self.eye_pupil_threshold,
            racial_feature_threshold=self.racial_feature_threshold
        )
        
        # 认识自己处理器
        self.self_recognition_handler = SelfRecognitionHandler(
            context=self.context,
            milvus_manager=self.milvus_manager,
            image_processor=self.image_processor
        )
        
        # 认识角色处理器
        self.character_recognition_handler = CharacterRecognitionHandler(
            context=self.context,
            milvus_manager=self.milvus_manager,
            image_processor=self.image_processor
        )
    
    # ==================== 指令：认识自己 ====================
    @filter.command("认识自己")
    async def teach_self(self, event: AstrMessageEvent):
        """教 AI 认识自己的形象"""
        event.stop_event()
        async for result in self.self_recognition_handler.handle_teach_self(event):
            yield result
    
    # ==================== 指令：认识角色 ====================
    @filter.command("认识角色")
    async def teach_character(self, event: AstrMessageEvent):
        """教 AI 认识其他角色的形象"""
        event.stop_event()
        async for result in self.character_recognition_handler.handle_teach_character(event):
            yield result
    
    # ==================== 指令：角色列表 ====================
    @filter.command("角色列表")
    async def list_characters(self, event: AstrMessageEvent):
        """查看已认识的角色列表"""
        event.stop_event()
        result = await self.character_recognition_handler.list_known_characters(event)
        yield event.plain_result(result)
    
    # ==================== 指令：设置查看 ====================
    @filter.command("角色认知设置")
    async def show_settings(self, event: AstrMessageEvent):
        """查看当前插件设置"""
        event.stop_event()
        
        if self.milvus_manager.use_custom_embedding:
            embedding_mode = f"自定义向量模型 (API: {self.milvus_manager.custom_embedding_api_base or '默认阿里云'}, 模型: {self.milvus_manager.custom_embedding_model}, 维度: {self.milvus_manager.custom_embedding_dim})"
        else:
            embedding_mode = "未配置向量模型 API Key"
        
        milvus_status = "✅ 已连接" if self.milvus_manager.is_connected() else f"❌ 未连接 ({self.milvus_manager.get_error()})"
        
        msg = f"""🧠 角色认知插件当前设置 (v2.1.1)：

📦 Milvus 状态: {milvus_status}
📍 地址: {self.milvus_manager.milvus_host}:{self.milvus_manager.milvus_port}
🗄️ 数据库: {self.milvus_manager.milvus_db_name}
📁 集合: {self.milvus_manager.collection_name}
🎯 向量相似度阈值: {self.similarity_threshold}
🎨 发色相似度阈值: {self.hair_color_threshold}
👁️ 眼白颜色阈值: {self.eye_white_threshold}
🔮 眼瞳颜色阈值: {self.eye_pupil_threshold}
🐱 种族特征阈值: {self.racial_feature_threshold}
👁️ 视觉模型提供商: {self.vision_provider_id or '未配置'}
🔤 向量模型模式: {embedding_mode}

使用方法：
• /认识自己 - 教 AI 认识自己的形象
• /认识角色 - 教 AI 认识其他角色
• /角色列表 - 查看已认识的角色
• /角色认知设置 - 查看当前配置
• 发送图片 - 自动识别图中是否有自己或认识的角色
"""
        yield event.plain_result(msg)
    
    # ==================== 自动识别图片 ====================
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_image_message(self, event: AstrMessageEvent):
        """处理图片消息，自动识别角色"""
        # 检查消息中是否有图片
        image_comps = [comp for comp in event.message_obj.message if isinstance(comp, Image)]
        if not image_comps:
            return
        
        event.stop_event()
        
        # 检查 Milvus 连接
        if not self.milvus_manager.is_connected():
            logger.warning("[角色认知] Milvus 未连接，跳过角色认知功能")
            return
        
        # 获取图片
        img_url_or_path = self.image_processor.get_image_url_from_component(image_comps[0])
        if not img_url_or_path:
            return
        
        # 解析图片数据
        image_data = await self.image_processor.resolve_image_data(img_url_or_path)
        if not image_data:
            await event.send(MessageChain([Plain("❌ 图片获取失败，请重试")]))
            return
        
        # 保存临时图片
        temp_file_path = self.image_processor.save_temp_image(image_data)
        if not temp_file_path:
            await event.send(MessageChain([Plain("❌ 图片处理失败")]))
            return
        
        try:
            # 读取图片并转为base64
            with open(temp_file_path, "rb") as f:
                img_bytes = f.read()
            ext = temp_file_path.suffix.lower()
            img_base64 = self.image_processor.image_bytes_to_base64(img_bytes, ext)
            
            session_id = event.unified_msg_origin
            user_text = event.message_str
            
            # 提前检测人物和提取特征，避免重复调用视觉模型
            has_person = await self.image_processor.check_if_image_has_person(img_base64)
            if not has_person:
                logger.info("[角色认知] 图片中未检测到人物，进行常规识图对话")
                await self._handle_normal_image_conversation(event, img_base64, user_text)
                return
            
            # 提取角色特征（只调用一次）
            feature_desc = await self.image_processor.analyze_image_for_recognition(img_base64)
            if not feature_desc:
                logger.info("[角色认知] 无法提取角色特征，进行常规识图对话")
                await self._handle_normal_image_conversation(event, img_base64, user_text)
                return
            
            # 第一步：检查是否是自己（传入预提取的特征）
            is_self, self_features = await self.self_recognition_handler.process_image_for_self(
                event, img_base64, session_id, feature_desc
            )
            
            if is_self:
                # 匹配到自己，生成自我认知回复
                reply = await self.self_recognition_handler.generate_self_response(
                    event, img_base64, self_features, user_text
                )
                await event.send(MessageChain([Plain(reply)]))
                # 存入对话历史，让AI后续对话能记住看过自己的图片
                user_msg_text = f"[图片消息] {user_text or '（用户发送了一张图片）'} 图片中检测到AI自己的形象，特征：{self_features}"
                await self._add_to_conversation_history(event, user_msg_text, reply)
                return
            
            # 第二步：检查是否是已认识的角色（传入预提取的特征）
            is_character, character_name, character_features = await self.character_recognition_handler.process_image_for_character(
                event, img_base64, session_id, feature_desc
            )
            
            if is_character:
                # 匹配到角色，生成角色认知回复
                reply = await self.character_recognition_handler.generate_character_response(
                    event, img_base64, character_name, character_features, user_text
                )
                await event.send(MessageChain([Plain(reply)]))
                # 存入对话历史，让AI后续对话能记住看过该角色的图片
                user_msg_text = f"[图片消息] {user_text or '（用户发送了一张图片）'} 图片中检测到角色{character_name}，特征：{character_features}"
                await self._add_to_conversation_history(event, user_msg_text, reply)
                return
            
            # 第三步：未匹配到任何已认知形象，进行常规识图对话
            logger.info("[角色认知] 未匹配到任何已认知形象，进行自然对话")
            await self._handle_normal_image_conversation(event, img_base64, user_text)
            
        except Exception as e:
            logger.error(f"[角色认知] 自动识别异常: {e}")
            await event.send(MessageChain([Plain("处理图片时出了点问题，稍后再试试吧。")]))
        finally:
            if temp_file_path:
                self.image_processor.cleanup_temp_file(temp_file_path)
    
    async def _get_recent_context(self, event: AstrMessageEvent, max_turns: int = 5) -> str:
        """
        获取最近N轮对话上下文
        
        Args:
            event: 消息事件
            max_turns: 最大轮数
            
        Returns:
            格式化的上下文文本
        """
        try:
            conv_mgr = self.context.conversationManager
            if not conv_mgr:
                logger.warning("[角色认知] 对话管理器不可用")
                return ""
            
            uid = event.unified_msg_origin
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
            if not curr_cid:
                return ""
            
            conversation = await conv_mgr.get_conversation(uid, curr_cid)
            if not conversation or not conversation.history:
                return ""
            
            # 解析历史记录
            try:
                history = json.loads(conversation.history) if isinstance(conversation.history, str) else conversation.history
            except json.JSONDecodeError:
                logger.warning("[角色认知] 历史记录解析失败")
                return ""
            
            if not isinstance(history, list):
                return ""
            
            # 提取最近的对话（每轮包含user和assistant两条消息）
            recent_messages = history[-(max_turns * 2):]
            
            context_parts = []
            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # 处理content可能是列表的情况
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = " ".join(text_parts)
                
                if role == "user":
                    context_parts.append(f"用户：{content}")
                elif role == "assistant":
                    context_parts.append(f"AI：{content}")
            
            if context_parts:
                return "\n".join(context_parts)
            return ""
            
        except Exception as e:
            logger.error(f"[角色认知] 获取上下文失败: {e}")
            return ""
    
    async def _add_to_conversation_history(self, event: AstrMessageEvent, user_msg_text: str, assistant_msg_text: str):
        """
        将识别结果和回复存入对话历史，解决AI不知道自己看过图片导致对话跳跃的问题
        
        Args:
            event: 消息事件
            user_msg_text: 用户消息文本（包含图片描述信息）
            assistant_msg_text: AI回复文本
        """
        try:
            conv_mgr = self.context.conversationManager
            if not conv_mgr:
                logger.warning("[角色认知] 对话管理器不可用，无法存入对话历史")
                return
            
            uid = event.unified_msg_origin
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
            if not curr_cid:
                logger.warning("[角色认知] 无法获取当前对话ID，跳过存入对话历史")
                return
            
            user_msg = UserMessageSegment(content=[TextPart(text=user_msg_text)])
            assistant_msg = AssistantMessageSegment(content=[TextPart(text=assistant_msg_text)])
            
            await conv_mgr.add_message_pair(
                cid=curr_cid,
                user_message=user_msg,
                assistant_message=assistant_msg,
            )
            logger.info("[角色认知] 已将识别结果和回复存入对话历史")
        except Exception as e:
            logger.error(f"[角色认知] 存入对话历史失败: {e}")
    
    async def _handle_normal_image_conversation(self, event: AstrMessageEvent, img_base64: str, user_text: str):
        """
        处理常规图片对话（未匹配到任何已认知形象时）
        
        Args:
            event: 消息事件
            img_base64: 图片base64编码
            user_text: 用户消息文本
        """
        try:
            provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
            
            # 分析图片内容
            image_analysis = await self.image_processor.analyze_image_content(img_base64)
            
            # 获取最近对话上下文
            recent_context = await self._get_recent_context(event, max_turns=5)
            
            # 构建系统提示词 - 模拟AI"看到"图片
            system_prompt = """你现在拥有视觉能力，可以看到用户展示给你的图片。

当前情况：你看到了一张图片，请以第一人称的视角自然地回应。

要求：
1. 把图片当作你自己"看到"的内容，而不是"用户发送的图片"
2. 自然地表达你对图片内容的观察和感受
3. 结合对话上下文进行回应
4. **关键要求：保持与对话上下文中一致的语言风格、格式和用语习惯**
   - 参考上下文中AI回复的语言风格（如是否使用括号描述动作、特殊的说话方式等）
   - 保持相同的语言特征，如句式结构、语气词、表达习惯
   - 让回复自然融入之前的对话风格中
5. 像是在和朋友聊天一样自然地回应"""

            # 构建用户消息
            context_section = ""
            if recent_context:
                context_section = f"""
【最近的对话上下文】
{recent_context}
"""

            enhanced_user_message = f"""{context_section}
【我现在看到的】
{image_analysis}

【用户对我说的话】
{user_text or '（用户只是给我看了这张图片）'}

请基于我看到的图片内容以及对话上下文，自然地回应。记住，你是以第一人称视角"看到"这张图片的。"""

            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=enhanced_user_message,
                system_prompt=system_prompt,
                conversation_id=event.unified_msg_origin
            )
            reply = llm_resp.completion_text
            await event.send(MessageChain([Plain(reply)]))
            # 存入对话历史，让AI后续对话能记住看过这张图片
            user_msg_text = f"[图片消息] {user_text or '（用户发送了一张图片）'} 图片内容：{image_analysis}"
            await self._add_to_conversation_history(event, user_msg_text, reply)
        except Exception as e:
            logger.error(f"[角色认知] LLM 生成回复失败: {e}")
            await event.send(MessageChain([Plain("处理图片时出了点问题，稍后再试试吧。")]))
    
    async def terminate(self):
        """插件卸载时断开 Milvus 连接"""
        self.milvus_manager.disconnect()
        logger.info("[角色认知] 插件已卸载")