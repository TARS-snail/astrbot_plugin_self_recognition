"""
认识自己模块
处理AI认识自己形象的功能
"""
import json
from pathlib import Path
from typing import Optional, List

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image, Plain
from astrbot.core.utils.session_waiter import session_waiter, SessionController

from .milvus_manager import MilvusManager
from .image_processor import ImageProcessor


class SelfRecognitionHandler:
    """认识自己处理器"""
    
    def __init__(self, context, milvus_manager: MilvusManager, image_processor: ImageProcessor):
        """
        初始化认识自己处理器
        
        Args:
            context: AstrBot Context
            milvus_manager: Milvus管理器
            image_processor: 图片处理器
        """
        self.context = context
        self.milvus_manager = milvus_manager
        self.image_processor = image_processor
    
    async def handle_teach_self(self, event: AstrMessageEvent):
        """
        处理 /认识自己 指令
        
        Args:
            event: 消息事件
        """
        temp_file_path = None
        try:
            # 检查 Milvus 连接
            if not self.milvus_manager.is_connected():
                error_msg = self.milvus_manager.get_error()
                yield event.plain_result(f"❌ Milvus 数据库连接失败，无法使用自我认知功能。\n错误: {error_msg}\n请检查插件配置中的 Milvus 认证信息。")
                return
            
            yield event.plain_result("📸 好的，请发送一张我的照片～（发送图片后我会记住我的样子，发送【取消】可退出）")
            
            @session_waiter(timeout=60)
            async def wait_for_image(controller: SessionController, img_event: AstrMessageEvent):
                nonlocal temp_file_path
                
                # 检查是否要取消
                user_text = img_event.message_str.strip()
                if user_text in ["取消", "退出", "结束", "停止", "cancel", "exit"]:
                    await img_event.send(MessageChain([Plain("✅ 已退出认识自己流程")]))
                    controller.stop()
                    return
                
                image_comps = [comp for comp in img_event.message_obj.message if isinstance(comp, Image)]
                if not image_comps:
                    await img_event.send(MessageChain([Plain("❌ 请发送一张图片，或发送【取消】退出")]))
                    controller.keep(timeout=60, reset_timeout=True)
                    return
                
                img_url_or_path = self.image_processor.get_image_url_from_component(image_comps[0])
                if not img_url_or_path:
                    await img_event.send(MessageChain([Plain("❌ 图片地址无效")]))
                    controller.keep(timeout=60, reset_timeout=True)
                    return
                
                image_data = await self.image_processor.resolve_image_data(img_url_or_path)
                if not image_data:
                    await img_event.send(MessageChain([Plain("❌ 图片获取失败")]))
                    controller.keep(timeout=60, reset_timeout=True)
                    return
                
                temp_file_path = self.image_processor.save_temp_image(image_data)
                if not temp_file_path:
                    await img_event.send(MessageChain([Plain("❌ 图片保存失败")]))
                    controller.keep(timeout=60, reset_timeout=True)
                    return
                
                await img_event.send(MessageChain([Plain("🔍 正在记住，请稍等...")]))
                
                with open(temp_file_path, "rb") as f:
                    img_bytes = f.read()
                ext = temp_file_path.suffix.lower()
                img_base64 = self.image_processor.image_bytes_to_base64(img_bytes, ext)
                
                # 分析图片特征
                feature_desc = await self.image_processor.analyze_self_image(img_base64)
                if not feature_desc:
                    await img_event.send(MessageChain([Plain("❌ 分析失败，请重试")]))
                    controller.keep(timeout=60, reset_timeout=True)
                    return
                
                # 提取颜色特征并缓存
                color_features = await self.image_processor.extract_color_features(feature_desc)
                session_id = img_event.unified_msg_origin
                if color_features.get("hair_color") or color_features.get("eye_color"):
                    self.milvus_manager.update_color_cache(session_id, color_features)
                
                # 存储到 Milvus
                memory_text = feature_desc
                success = await self.milvus_manager.add_memory(
                    text=memory_text,
                    session_id=session_id,
                    character_type="self",
                    character_name=""
                )
                
                if success:
                    await img_event.send(MessageChain([Plain(f"✅ 认识啦！我的形象特征：{feature_desc}")]))
                    controller.stop()
                else:
                    await img_event.send(MessageChain([Plain("❌ 存储失败")]))
                    controller.keep(timeout=60, reset_timeout=True)
            
            await wait_for_image(event)
        except TimeoutError:
            yield event.plain_result("⏰ 操作超时")
        except Exception as e:
            logger.error(f"[角色认知] 认识自己异常: {e}")
            yield event.plain_result(f"❌ 出错: {str(e)}")
        finally:
            if temp_file_path:
                self.image_processor.cleanup_temp_file(temp_file_path)
    
    async def process_image_for_self(self, event: AstrMessageEvent, img_base64: str, session_id: str, feature_desc: str = None):
        """
        处理图片，检测是否包含自己的形象
        
        Args:
            event: 消息事件
            img_base64: 图片base64编码
            session_id: 会话ID
            feature_desc: 预提取的角色特征描述（可选，如果不提供则自动提取）
            
        Returns:
            tuple: (is_self: bool, reply: str or None)
        """
        try:
            # 如果没有提供预提取的特征，则自动提取
            if not feature_desc:
                # 检测图片中是否有人物
                has_person = await self.image_processor.check_if_image_has_person(img_base64)
                if not has_person:
                    logger.info("[角色认知] 图片中未检测到人物")
                    return False, None
                
                # 分析图片中角色的特征
                feature_desc = await self.image_processor.analyze_image_for_recognition(img_base64)
                if not feature_desc:
                    return False, "🔍 看不太清图片里的内容呢，能再发一张清楚的吗？"
            
            # 搜索记忆
            memories = await self.milvus_manager.search_memory(
                query_text=feature_desc,
                session_id=session_id,
                character_type="self",
                top_k=1
            )
            
            # 颜色特征检查
            color_check_passed = True
            if memories and memories[0]["score"] >= self.image_processor.similarity_threshold:
                # 提取新图片的颜色特征
                current_color_features = await self.image_processor.extract_color_features(feature_desc)
                # 获取存储的颜色特征
                stored_color_features = self.milvus_manager.get_color_cache(session_id)
                
                # 如果缓存为空，从数据库存储的记忆文本中提取颜色特征
                if not stored_color_features and memories[0].get("text"):
                    stored_color_features = await self.image_processor.extract_color_features(memories[0]["text"])
                    if stored_color_features:
                        self.milvus_manager.update_color_cache(session_id, stored_color_features)
                        logger.info(f"[角色认知] 从数据库记忆中提取颜色特征: {stored_color_features}")
                
                logger.info(f"[角色认知] 颜色特征检查: 当前 {current_color_features}, 存储 {stored_color_features}")
                
                # 如果存储的颜色特征仍然为空，跳过颜色检查（无法验证）
                if not stored_color_features:
                    logger.warning("[角色认知] 无法获取存储的颜色特征，跳过颜色验证")
                else:
                    # 检查发色相似度
                    hair_similarity = 1.0
                    if current_color_features.get("hair_color") and stored_color_features.get("hair_color"):
                        hair_similarity = self.image_processor.calculate_color_similarity(
                            current_color_features["hair_color"],
                            stored_color_features["hair_color"]
                        )
                    
                    # 检查眼白颜色相似度
                    eye_white_similarity = 1.0
                    if current_color_features.get("eye_white_color") and stored_color_features.get("eye_white_color"):
                        eye_white_similarity = self.image_processor.calculate_color_similarity(
                            current_color_features["eye_white_color"],
                            stored_color_features["eye_white_color"]
                        )
                    
                    # 检查眼瞳颜色相似度
                    eye_pupil_similarity = 1.0
                    if current_color_features.get("eye_pupil_color") and stored_color_features.get("eye_pupil_color"):
                        eye_pupil_similarity = self.image_processor.calculate_color_similarity(
                            current_color_features["eye_pupil_color"],
                            stored_color_features["eye_pupil_color"]
                        )
                    # 兼容旧格式
                    elif current_color_features.get("eye_color") and stored_color_features.get("eye_color"):
                        eye_pupil_similarity = self.image_processor.calculate_color_similarity(
                            current_color_features["eye_color"],
                            stored_color_features["eye_color"]
                        )
                    
                    # 检查种族特征相似度
                    racial_similarity = 1.0
                    current_racial = current_color_features.get("racial_features", "")
                    stored_racial = stored_color_features.get("racial_features", "")
                    if current_racial or stored_racial:
                        # 如果一方有种族特征，另一方没有，需要检查
                        if (current_racial and not stored_racial) or (not current_racial and stored_racial):
                            # 检查是否为"无"或空
                            if current_racial not in ["无", ""] or stored_racial not in ["无", ""]:
                                racial_similarity = 0.0
                        elif current_racial and stored_racial:
                            # 双方都有种族特征，计算相似度
                            if current_racial not in ["无", ""] and stored_racial not in ["无", ""]:
                                racial_similarity = self.image_processor.calculate_feature_similarity(
                                    current_racial,
                                    stored_racial
                                )
                    
                    logger.info(f"[角色认知] 特征相似度: 发色 {hair_similarity:.2f}, 眼白 {eye_white_similarity:.2f}, 眼瞳 {eye_pupil_similarity:.2f}, 种族 {racial_similarity:.2f}")
                    
                    # 如果任何特征相似度低于阈值，判定为不是自己
                    if hair_similarity < self.image_processor.hair_color_threshold:
                        logger.info(f"[角色认知] 发色不匹配，判定为不是自己")
                        color_check_passed = False
                    elif eye_white_similarity < self.image_processor.eye_white_threshold:
                        logger.info(f"[角色认知] 眼白颜色不匹配，判定为不是自己")
                        color_check_passed = False
                    elif eye_pupil_similarity < self.image_processor.eye_pupil_threshold:
                        logger.info(f"[角色认知] 眼瞳颜色不匹配，判定为不是自己")
                        color_check_passed = False
                    elif racial_similarity < self.image_processor.racial_feature_threshold:
                        logger.info(f"[角色认知] 种族特征不匹配，判定为不是自己")
                        color_check_passed = False
            
            # 判断是否匹配成功
            if color_check_passed and memories and memories[0]["score"] >= self.image_processor.similarity_threshold:
                logger.info(f"[角色认知] 自我匹配成功！相似度: {memories[0]['score']}")
                return True, memories[0]["text"]
            else:
                logger.info("[角色认知] 未匹配到自己的形象")
                return False, None
                
        except Exception as e:
            logger.error(f"[角色认知] 自我识别异常: {e}")
            return False, None
    
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
            conv_mgr = self.context.conversation_manager
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
    
    async def generate_self_response(
        self, 
        event: AstrMessageEvent, 
        img_base64: str, 
        self_features: str,
        user_text: str
    ) -> str:
        """
        生成认识自己的回复
        
        Args:
            event: 消息事件
            img_base64: 图片base64编码
            self_features: 自己的特征描述
            user_text: 用户消息文本
            
        Returns:
            回复文本
        """
        try:
            # 获取主模型提供商ID
            provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
            
            # 详细分析图片内容
            image_description = await self.image_processor.analyze_image_detail(img_base64)
            
            # 获取最近对话上下文
            recent_context = await self._get_recent_context(event, max_turns=5)
            
            # 构建系统提示词 - 模拟AI"看到"图片
            system_prompt = """你现在拥有视觉能力，可以看到用户展示给你的图片。

当前情况：你看到了一张图片，图片中的人物就是你自己。你需要以第一人称的视角自然地回应。

要求：
1. 把图片当作你自己"看到"的内容，而不是"用户发送的图片"
2. 自然地表达你看到图片中自己的反应
3. 结合你的形象特征和图片中的场景、动作、表情进行回应
4. **关键要求：保持与对话上下文中一致的语言风格、格式和用语习惯**
   - 参考上下文中AI回复的语言风格（如是否使用括号描述动作、特殊的说话方式等）
   - 保持相同的语言特征，如句式结构、语气词、表达习惯
   - 让回复自然融入之前的对话风格中
5. 保持自然的语气，像是在和朋友聊天"""

            # 构建用户消息
            context_section = ""
            if recent_context:
                context_section = f"""
【最近的对话上下文】
{recent_context}
"""

            enhanced_user_message = f"""{context_section}
【我现在看到的】
{image_description}

【我的形象特征】
{self_features}

【用户对我说的话】
{user_text or '（用户只是给我看了这张图片）'}

请基于我看到的图片内容、我的形象特征以及对话上下文，自然地回应。记住，图片中的人物就是我自己，我是以第一人称视角"看到"这张图片的。"""

            # 调用主模型
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=enhanced_user_message,
                system_prompt=system_prompt,
                conversation_id=event.unified_msg_origin
            )
            return llm_resp.completion_text
        except Exception as e:
            logger.error(f"[角色认知] LLM 生成回复失败: {e}")
            return "唔，我想说什么来着……能再说一遍吗？"
