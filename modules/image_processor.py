"""
图片处理模块
负责图片分析、特征提取、颜色处理等功能
"""
import os
import base64
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from astrbot.api import logger
from astrbot.api.message_components import Image


class ImageProcessor:
    """图片处理器"""
    
    def __init__(self, context, vision_provider_id: str, similarity_threshold: float = 0.65, 
                 hair_color_threshold: float = 0.8, eye_white_threshold: float = 0.8,
                 eye_pupil_threshold: float = 0.8, racial_feature_threshold: float = 0.8):
        """
        初始化图片处理器
        
        Args:
            context: AstrBot Context
            vision_provider_id: 视觉模型提供商ID
            similarity_threshold: 向量相似度阈值
            hair_color_threshold: 发色相似度阈值
            eye_white_threshold: 眼白颜色相似度阈值
            eye_pupil_threshold: 眼瞳颜色相似度阈值
            racial_feature_threshold: 种族特征相似度阈值
        """
        self.context = context
        self.vision_provider_id = vision_provider_id
        self.similarity_threshold = similarity_threshold
        self.hair_color_threshold = hair_color_threshold
        self.eye_white_threshold = eye_white_threshold
        self.eye_pupil_threshold = eye_pupil_threshold
        self.racial_feature_threshold = racial_feature_threshold
        
        # 临时目录
        self.temp_dir = Path(__file__).parent.parent / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def analyze_self_image(self, image_base64: str) -> str:
        """
        分析自己的形象图片，提取特征
        
        Args:
            image_base64: 图片的 base64 编码
            
        Returns:
            特征描述文本
        """
        if not self.vision_provider_id:
            raise ValueError("未配置视觉模型提供商")
        
        prompt = """请描述这张二次元图片中角色的核心识别特征，按以下格式输出：

【发色】（如金色、黑色、蓝色、粉色、银色等，如有渐变或挑染请说明）
【发型】（如双马尾、单麻花辫、披肩发、短发、长发、刘海样式等）
【眼白颜色】（通常是白色，特殊角色可能是其他颜色）
【眼瞳颜色】（虹膜颜色，如紫色、蓝色、红色等，异色瞳请说明左右眼颜色）
【呆毛】（有或没有，如有请描述位置和形状）
【种族特征】（非人类特征，如猫耳、狗耳、兔耳、龙角、恶魔角、尾巴、翅膀等，没有则填"无"）

请用简洁的语言描述，确保不同画师绘制的同一角色都能识别。只返回以上格式的特征描述，不要有其他内容，忽略服饰、面部表情、姿势等其他特征。"""
        
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt,
                image_urls=[image_base64]
            )
            description = resp.completion_text.strip()
            logger.info(f"[角色认知] 视觉分析结果: {description[:100]}...")
            return description
        except Exception as e:
            logger.error(f"[角色认知] 视觉模型调用失败: {e}")
            return ""
    
    async def analyze_character_image(self, image_base64: str, character_name: str = "") -> str:
        """
        分析角色形象图片，提取特征
        
        Args:
            image_base64: 图片的 base64 编码
            character_name: 角色名称（可选）
            
        Returns:
            特征描述文本
        """
        if not self.vision_provider_id:
            raise ValueError("未配置视觉模型提供商")
        
        name_hint = f"（角色名：{character_name}）" if character_name else ""
        prompt = f"""请描述这张二次元图片中角色的核心识别特征{name_hint}，按以下格式输出：

【发色】（如金色、黑色、蓝色、粉色、银色等，如有渐变或挑染请说明）
【发型】（如双马尾、单麻花辫、披肩发、短发、长发、刘海样式等）
【眼白颜色】（通常是白色，特殊角色可能是其他颜色）
【眼瞳颜色】（虹膜颜色，如紫色、蓝色、红色等，异色瞳请说明左右眼颜色）
【呆毛】（有或没有，如有请描述位置和形状）
【种族特征】（非人类特征，如猫耳、狗耳、兔耳、龙角、恶魔角、尾巴、翅膀等，没有则填"无"）

请用简洁的语言描述，确保不同画师绘制的同一角色都能识别。只返回以上格式的特征描述，不要有其他内容，忽略服饰、面部表情、姿势等其他特征。"""
        
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt,
                image_urls=[image_base64]
            )
            description = resp.completion_text.strip()
            logger.info(f"[角色认知] 角色视觉分析结果: {description[:100]}...")
            return description
        except Exception as e:
            logger.error(f"[角色认知] 角色视觉模型调用失败: {e}")
            return ""
    
    async def check_if_image_has_person(self, image_base64: str) -> bool:
        """
        检测图片中是否有人物或角色
        
        Args:
            image_base64: 图片的 base64 编码
            
        Returns:
            是否检测到人物/角色
        """
        if not self.vision_provider_id:
            raise ValueError("未配置视觉模型提供商")
        
        prompt = "这张图片中是否有人物或动漫角色？请只回答'是'或'否'，不要回答其他内容。"
        
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt,
                image_urls=[image_base64]
            )
            answer = resp.completion_text.strip()
            has_person = "是" in answer
            logger.info(f"[角色认知] 人物检测结果: {answer} -> {has_person}")
            return has_person
        except Exception as e:
            logger.error(f"[角色认知] 人物检测失败: {e}")
            return False
    
    async def analyze_image_for_recognition(self, image_base64: str) -> str:
        """
        分析图片用于识别匹配
        
        Args:
            image_base64: 图片的 base64 编码
            
        Returns:
            特征描述文本
        """
        if not self.vision_provider_id:
            raise ValueError("未配置视觉模型提供商")
        
        prompt = """请描述这张二次元图片中角色的核心识别特征，按以下格式输出：

【发色】（如金色、黑色、蓝色、粉色、银色等，如有渐变或挑染请说明）
【发型】（如双马尾、单麻花辫、披肩发、短发、长发、刘海样式等）
【眼白颜色】（通常是白色，特殊角色可能是其他颜色）
【眼瞳颜色】（虹膜颜色，如紫色、蓝色、红色等，异色瞳请说明左右眼颜色）
【呆毛】（有或没有，如有请描述位置和形状）
【种族特征】（非人类特征，如猫耳、狗耳、兔耳、龙角、恶魔角、尾巴、翅膀等，没有则填"无"）

请用简洁的语言描述，确保不同画师绘制的同一角色都能识别。只返回以上格式的特征描述，不要有其他内容，忽略服饰、面部表情、姿势等其他特征。"""
        
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt,
                image_urls=[image_base64]
            )
            description = resp.completion_text.strip()
            logger.info(f"[角色认知] 识别分析结果: {description[:100]}...")
            return description
        except Exception as e:
            logger.error(f"[角色认知] 识别分析失败: {e}")
            return ""
    
    async def analyze_image_content(self, image_base64: str) -> str:
        """
        分析图片内容（用于自然对话）
        
        Args:
            image_base64: 图片的 base64 编码
            
        Returns:
            图片内容描述
        """
        if not self.vision_provider_id:
            raise ValueError("未配置视觉模型提供商")
        
        prompt = """用户发送了一张图片。请描述图片的主要内容，用于后续对话理解。请根据图片内容进行简要分析。"""
        
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt,
                image_urls=[image_base64]
            )
            description = resp.completion_text.strip()
            logger.info(f"[角色认知] 图片内容分析: {description[:100]}...")
            return description
        except Exception as e:
            logger.error(f"[角色认知] 图片内容分析失败: {e}")
            return ""
    
    async def analyze_image_detail(self, image_base64: str) -> str:
        """
        详细分析图片内容（用于匹配成功时的详细描述）
        
        Args:
            image_base64: 图片的 base64 编码
            
        Returns:
            详细图片描述
        """
        if not self.vision_provider_id:
            raise ValueError("未配置视觉模型提供商")
        
        prompt = """请详细描述这张图片的内容，包括：
1. 场景和环境（在哪里、背景是什么）
2. 人物的动作和姿态（正在做什么）
3. 人物的表情和神态
4. 其他重要细节

请客观描述，不要遗漏重要信息。"""
        
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt,
                image_urls=[image_base64]
            )
            description = resp.completion_text.strip()
            logger.info(f"[角色认知] 图片详细分析: {description[:100]}...")
            return description
        except Exception as e:
            logger.error(f"[角色认知] 图片详细分析失败: {e}")
            return ""
    
    async def extract_color_features(self, feature_desc: str) -> Dict[str, str]:
        """
        从特征描述中提取完整特征（发色、眼白颜色、眼瞳颜色、种族特征）
        
        Args:
            feature_desc: 角色特征描述文本
            
        Returns:
            {"hair_color": "银白色", "eye_white_color": "白色", "eye_pupil_color": "紫色", "racial_features": "猫耳,尾巴"} 或 {}
        """
        if not feature_desc:
            return {}
        
        # 使用视觉模型提取结构化特征信息
        result = {
            "hair_color": "",
            "eye_white_color": "",
            "eye_pupil_color": "",
            "racial_features": ""
        }
        
        try:
            prompt = f"""根据以下角色描述，提取角色的各项特征：

描述：
{feature_desc}

请用以下JSON格式返回（注意区分眼白颜色和眼瞳颜色）：
{{
    "hair_color": "发色（如银白色、黑色等，如有渐变或挑染请完整描述）",
    "eye_white_color": "眼白颜色（通常是白色，特殊角色可能是其他颜色）",
    "eye_pupil_color": "眼瞳/虹膜颜色（如紫色、蓝色等，异色瞳请说明左右眼颜色）",
    "racial_features": "种族特征（非人类特征，如猫耳、龙角、尾巴等，没有则填'无'）"
}}

如果无法确定某项特征，使用空字符串""
"""
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt
            )
            import json
            try:
                json_str = resp.completion_text.strip()
                # 提取JSON部分
                if "{" in json_str and "}" in json_str:
                    json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
                extracted = json.loads(json_str)
                
                # 兼容旧格式
                if extracted.get("eye_color") and not extracted.get("eye_pupil_color"):
                    extracted["eye_pupil_color"] = extracted["eye_color"]
                
                if extracted.get("hair_color"):
                    result["hair_color"] = extracted["hair_color"]
                if extracted.get("eye_white_color"):
                    result["eye_white_color"] = extracted["eye_white_color"]
                if extracted.get("eye_pupil_color"):
                    result["eye_pupil_color"] = extracted["eye_pupil_color"]
                if extracted.get("racial_features"):
                    result["racial_features"] = extracted["racial_features"]
                    
            except Exception as e:
                logger.warning(f"[角色认知] JSON解析失败: {e}")
                # 降级到简单的颜色提取逻辑
                result = await self._extract_color_features_simple(feature_desc)
        except Exception as e:
            logger.warning(f"[角色认知] 颜色特征提取失败: {e}")
            # 降级到简单的颜色提取逻辑
            result = await self._extract_color_features_simple(feature_desc)
        
        logger.info(f"[角色认知] 特征提取结果: {result}")
        return result
    
    async def _extract_color_features_simple(self, feature_desc: str) -> Dict[str, str]:
        """
        简单的特征提取逻辑（降级方案）
        
        Args:
            feature_desc: 角色特征描述文本
            
        Returns:
            特征字典
        """
        result = {
            "hair_color": "",
            "eye_white_color": "",
            "eye_pupil_color": "",
            "racial_features": ""
        }
        desc_lower = feature_desc.lower()
        
        # 颜色词汇
        colors = {
            "金": ["金色", "金黄色", "金发", "淡金", "浅金"],
            "银": ["银色", "银白色", "银发", "银灰"],
            "黑": ["黑色", "乌黑", "漆黑", "黑发"],
            "白": ["白色", "白发", "雪白"],
            "蓝": ["蓝色", "天蓝", "蔚蓝", "蓝发", "蓝眼"],
            "红": ["红色", "绯红", "红发", "赤红"],
            "粉": ["粉色", "粉红", "粉发"],
            "紫": ["紫色", "紫发", "紫眼", "深紫"],
            "绿": ["绿色", "绿发", "绿眼", "碧绿"],
            "棕": ["棕色", "棕发", "褐发", "咖啡色"],
            "灰": ["灰色", "灰发", "银灰"]
        }
        
        # 种族特征关键词
        racial_keywords = [
            "猫耳", "狗耳", "兔耳", "狐耳", "兽耳",
            "龙角", "恶魔角", "独角", "角",
            "尾巴", "猫尾", "狐尾", "兔尾",
            "翅膀", "天使翼", "恶魔翼",
            "兽人", "人鱼", "精灵", "妖精"
        ]
        
        # 提取发色
        for color_key, color_variants in colors.items():
            for variant in color_variants:
                if variant in desc_lower:
                    if "发" in variant or "发色" in desc_lower:
                        result["hair_color"] = variant
                        break
                    elif not result["hair_color"]:
                        result["hair_color"] = variant
        
        # 提取眼瞳颜色
        eye_keywords = ["瞳", "瞳色", "虹膜", "眼瞳"]
        for color_key, color_variants in colors.items():
            for variant in color_variants:
                if variant in desc_lower:
                    variant_pos = desc_lower.find(variant)
                    if variant_pos != -1:
                        start = max(0, variant_pos - 30)
                        end = min(len(desc_lower), variant_pos + 30)
                        context = desc_lower[start:end]
                        if any(keyword in context for keyword in eye_keywords):
                            result["eye_pupil_color"] = variant
                            break
        
        # 默认眼白颜色为白色
        result["eye_white_color"] = "白色"
        
        # 提取种族特征
        found_features = []
        for keyword in racial_keywords:
            if keyword in desc_lower:
                found_features.append(keyword)
        if found_features:
            result["racial_features"] = ",".join(found_features)
        
        # 兼容旧格式：如果没有提取到眼瞳颜色，尝试从 eye_color 字段提取
        if not result["eye_pupil_color"] and "眼" in desc_lower:
            for color_key, color_variants in colors.items():
                for variant in color_variants:
                    if variant in desc_lower:
                        result["eye_pupil_color"] = variant
                        break
                if result["eye_pupil_color"]:
                    break
        
        return result
    
    def calculate_feature_similarity(self, feature1: str, feature2: str) -> float:
        """
        计算两个特征字符串的相似度 (0-1)
        支持种族特征的多特征对比（如"猫耳,尾巴"）
        
        Args:
            feature1: 特征1
            feature2: 特征2
            
        Returns:
            相似度 (0-1)
        """
        if not feature1 or not feature2:
            return 0.0
        
        feature1_lower = feature1.lower()
        feature2_lower = feature2.lower()
        
        # 完全相同
        if feature1_lower == feature2_lower:
            return 1.0
        
        # 对于种族特征，检查是否有重叠的特征
        if "," in feature1_lower or "," in feature2_lower:
            features1 = set(f.strip() for f in feature1_lower.split(","))
            features2 = set(f.strip() for f in feature2_lower.split(","))
            
            # 计算交集比例
            intersection = features1 & features2
            union = features1 | features2
            
            if not union:
                return 0.0
            
            # 如果有交集，返回较高相似度
            if intersection:
                return len(intersection) / len(union)
        
        # 检查是否包含关系
        if feature1_lower in feature2_lower or feature2_lower in feature1_lower:
            return 0.8
        
        # 种族特征相似映射
        similar_features = [
            (["猫耳", "兽耳"], 0.7),
            (["狗耳", "兽耳"], 0.7),
            (["兔耳", "兽耳"], 0.7),
            (["狐耳", "兽耳"], 0.7),
            (["龙角", "角"], 0.8),
            (["恶魔角", "角"], 0.8),
            (["猫尾", "尾巴"], 0.8),
            (["狐尾", "尾巴"], 0.8),
            (["兔尾", "尾巴"], 0.8),
        ]
        
        for group, similarity in similar_features:
            if any(f in feature1_lower for f in group) and any(f in feature2_lower for f in group):
                return similarity
        
        return 0.0
    
    def calculate_color_similarity(self, color1: str, color2: str) -> float:
        """
        计算两个颜色的相似度 (0-1)
        
        Args:
            color1: 颜色1
            color2: 颜色2
            
        Returns:
            相似度 (0-1)
        """
        if not color1 or not color2:
            return 0.0
        
        color1_lower = color1.lower()
        color2_lower = color2.lower()
        
        # 完全相同
        if color1_lower == color2_lower:
            return 1.0
        
        # 颜色映射到主要色系
        color_families = {
            "金": ["金色", "金黄色", "金发", "淡金", "浅金", "黄金"],
            "银": ["银色", "银白色", "银发", "银灰", "白银", "灰银"],
            "黑": ["黑色", "乌黑", "漆黑", "黑发", "乌黑", "纯黑"],
            "白": ["白色", "白发", "雪白", "纯白", "洁白"],
            "蓝": ["蓝色", "天蓝", "蔚蓝", "蓝发", "蓝眼", "海蓝", "深蓝", "浅蓝"],
            "红": ["红色", "绯红", "红发", "赤红", "鲜红", "深红", "浅红"],
            "粉": ["粉色", "粉红", "粉发", "淡粉", "桃粉", "玫瑰粉"],
            "紫": ["紫色", "紫发", "紫眼", "深紫", "淡紫", "浅紫", "紫罗兰"],
            "绿": ["绿色", "绿发", "绿眼", "碧绿", "深绿", "浅绿", "草绿"],
            "棕": ["棕色", "棕发", "褐发", "咖啡色", "褐色", "茶色", "栗色"],
            "灰": ["灰色", "灰发", "银灰", "深灰", "浅灰", "灰白"]
        }
        
        # 查找颜色所属的色系
        family1 = None
        family2 = None
        
        for family, variants in color_families.items():
            for variant in variants:
                if variant in color1_lower:
                    family1 = family
                    break
            if family1:
                break
        
        for family, variants in color_families.items():
            for variant in variants:
                if variant in color2_lower:
                    family2 = family
                    break
            if family2:
                break
        
        # 同一色系，高相似度
        if family1 and family2 and family1 == family2:
            return 0.85
        
        # 相关色系（如金-棕，蓝-紫等）
        related_groups = [("金", "棕"), ("蓝", "紫"), ("红", "粉"), ("黑", "灰"), ("白", "银")]
        for group in related_groups:
            if (family1 == group[0] and family2 == group[1]) or (family1 == group[1] and family2 == group[0]):
                return 0.7
        
        # 完全不相关
        return 0.3
    
    # ==================== 图片处理工具方法 ====================
    
    async def resolve_image_data(self, img_url_or_path: str) -> Optional[bytes]:
        """
        解析图片数据
        
        Args:
            img_url_or_path: 图片URL或路径
            
        Returns:
            图片字节数据
        """
        # file:// 协议
        if img_url_or_path.startswith("file://"):
            file_path = img_url_or_path[7:]
            while file_path.startswith('/'):
                file_path = file_path[1:]
            file_path = '/' + file_path if not file_path.startswith('/') else file_path
            if Path(file_path).exists():
                try:
                    with open(file_path, "rb") as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"[角色认知] 读取本地文件失败: {e}")
                    return None
            return None
        
        # 本地路径
        file_path = Path(img_url_or_path)
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"[角色认知] 读取本地文件失败: {e}")
                return None
        
        # HTTP URL
        if img_url_or_path.startswith(("http://", "https://")):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(img_url_or_path, timeout=30) as resp:
                        if resp.status == 200:
                            return await resp.read()
                        else:
                            logger.error(f"[角色认知] 下载失败 HTTP {resp.status}")
                            return None
            except Exception as e:
                logger.error(f"[角色认知] 下载异常: {e}")
                return None
        return None
    
    def image_bytes_to_base64(self, image_data: bytes, ext: str = ".jpg") -> str:
        """
        将图片字节数据转换为 base64 编码
        
        Args:
            image_data: 图片字节数据
            ext: 文件扩展名
            
        Returns:
            base64 编码的图片字符串
        """
        base64_str = base64.b64encode(image_data).decode("utf-8")
        mime = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"
        return f"data:{mime};base64,{base64_str}"
    
    def save_temp_image(self, image_data: bytes) -> Optional[Path]:
        """
        保存临时图片
        
        Args:
            image_data: 图片字节数据
            
        Returns:
            临时文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"character_recog_{timestamp}.jpg"
            filepath = self.temp_dir / filename
            with open(filepath, "wb") as f:
                f.write(image_data)
            logger.info(f"[角色认知] 临时图片已保存: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"[角色认知] 保存临时图片失败: {e}")
            return None
    
    def cleanup_temp_file(self, filepath: Path):
        """
        清理临时文件
        
        Args:
            filepath: 临时文件路径
        """
        try:
            if filepath.exists():
                filepath.unlink()
                logger.info(f"[角色认知] 临时文件已删除: {filepath}")
        except Exception as e:
            logger.error(f"[角色认知] 删除临时文件失败: {e}")
    
    def get_image_url_from_component(self, comp: Image) -> Optional[str]:
        """
        从消息组件中获取图片URL
        
        Args:
            comp: 图片消息组件
            
        Returns:
            图片URL或路径
        """
        if hasattr(comp, 'url') and comp.url:
            return comp.url
        if hasattr(comp, 'file') and comp.file:
            return comp.file
        return None