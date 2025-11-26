"""
教师模型封装 - DeepSeek R1
通过Ollama API进行交互
"""

import ollama
import torch
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class TeacherResponse:
    """教师模型响应封装"""
    text: str
    embeddings: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None

class TeacherModel:
    """DeepSeek R1 教师模型封装"""
    
    def __init__(self, model_name: str = "deepseek-r1", temperature: float = 0.7):
        """
        初始化教师模型
        
        Args:
            model_name: Ollama中的模型名称
            temperature: 生成温度
        """
        self.model_name = model_name
        self.temperature = temperature
        self.client = ollama.Client()
        self._check_model_availability()
        
    def _check_model_availability(self):
        """检查模型是否在Ollama中可用"""
        try:
            models = self.client.list()
            available_models = [m.model for m in models['models']]
            if self.model_name not in available_models:
                raise ValueError(f"模型 {self.model_name} 未在Ollama中找到。可用模型: {available_models}")
            logger.info(f"教师模型 {self.model_name} 已就绪")
        except Exception as e:
            logger.error(f"检查模型可用性失败: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> TeacherResponse:
        """
        生成教师模型的响应
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            
        Returns:
            TeacherResponse: 教师模型响应
        """
        try:
            # 构建药学知识问答的专用提示
            pharma_prompt = self._build_pharma_prompt(prompt)
            
            response = self.client.generate(
                model=self.model_name,
                prompt=pharma_prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'stop': ['人类:', '助手:', '</s>']
                }
            )
            
            # 提取响应内容
            generated_text = response['response'].strip()
            
            # 计算置信度（基于响应长度和连贯性）
            confidence = self._calculate_confidence(generated_text)
            
            logger.info(f"教师模型生成响应: {generated_text[:100]}...")
            
            return TeacherResponse(
                text=generated_text,
                confidence=confidence,
                metadata={
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'prompt_length': len(prompt),
                    'response_length': len(generated_text)
                }
            )
            
        except Exception as e:
            logger.error(f"教师模型生成失败: {e}")
            return TeacherResponse(
                text="抱歉，我无法回答这个问题。",
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _build_pharma_prompt(self, user_input: str) -> str:
        """构建药学知识专用提示"""
        system_prompt = """你是一位专业的药学专家，拥有丰富的药物知识和临床经验。
        请基于你的专业知识，准确、清晰地回答以下药学相关问题。
        回答应包含：
        1. 直接回答问题的核心内容
        2. 相关的药理机制解释（如适用）
        3. 用药注意事项和副作用提醒（如适用）
        4. 建议的用药方案或处理建议
        
        请保持回答的专业性和准确性，响应长度控制在2000字符以内。"""
        
        return f"{system_prompt}\n\n人类: {user_input}\n\n助手:"
    
    def _calculate_confidence(self, text: str) -> float:
        """基于文本特征计算置信度"""
        if not text or len(text) < 10:
            return 0.0
        
        # 基于长度的基础置信度
        length_score = min(len(text) / 200, 1.0)
        
        # 检查是否有不确定性词汇
        uncertainty_words = ['可能', '或许', '大概', '不确定', '不清楚', '不知道']
        uncertainty_penalty = sum(1 for word in uncertainty_words if word in text) * 0.1
        
        # 检查是否有专业术语（提高置信度）
        pharma_terms = ['药物', '治疗', '剂量', '副作用', '药理', '临床', '用药']
        pharma_bonus = sum(1 for term in pharma_terms if term in text) * 0.05
        
        confidence = max(0.0, min(1.0, length_score - uncertainty_penalty + pharma_bonus))
        return confidence
    
    def batch_generate(self, prompts: List[str], max_tokens: int = 2048) -> List[TeacherResponse]:
        """批量生成响应"""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, max_tokens)
            responses.append(response)
            time.sleep(0.1)  # 避免API调用过于频繁
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.model_name,
            'temperature': self.temperature,
            'type': 'teacher_model',
            'description': 'DeepSeek R1 作为教师模型，提供高质量的药学知识蒸馏信号'
        }