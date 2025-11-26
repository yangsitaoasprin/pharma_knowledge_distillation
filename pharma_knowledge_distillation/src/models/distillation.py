"""
知识蒸馏核心引擎
实现教师-学生模型的知识迁移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from .teacher_model import TeacherModel, TeacherResponse
from .student_model import StudentModel, StudentResponse

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """知识蒸馏配置"""
    temperature: float = 3.0  # 蒸馏温度
    alpha: float = 0.7       # 硬标签损失权重
    beta: float = 0.3        # 软标签损失权重
    gamma: float = 0.1       # 特征蒸馏权重
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 4
    save_interval: int = 5
    eval_interval: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "temperature": self.temperature,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "save_interval": self.save_interval,
            "eval_interval": self.eval_interval,
        }

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            targets: 真实标签
            
        Returns:
            Dict[str, torch.Tensor]: 各种损失组件
        """
        # 硬标签损失（学生 vs 真实标签）
        hard_loss = self.ce_loss(student_logits, targets)
        
        # 软标签损失（学生 vs 教师）
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_predictions, soft_targets) * (self.temperature ** 2)
        
        # 组合损失
        total_loss = self.alpha * hard_loss + self.beta * soft_loss
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss
        }

class KnowledgeDistillationEngine:
    """知识蒸馏引擎"""
    
    def __init__(self, teacher_model: TeacherModel, student_model: StudentModel, 
                 config: DistillationConfig):
        """
        初始化知识蒸馏引擎
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            config: 蒸馏配置
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.loss_function = DistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            beta=config.beta
        )
        self.training_history = []
        self.evaluation_history = []
        
        # 创建输出目录
        self.output_dir = f"outputs/distillation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"知识蒸馏引擎初始化完成，输出目录: {self.output_dir}")
    
    def prepare_pharma_knowledge(self, knowledge_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备药学知识数据用于蒸馏
        
        Args:
            knowledge_data: 原始药学知识数据
            
        Returns:
            List[Dict[str, Any]]: 处理后的蒸馏数据
        """
        distilled_data = []
        
        logger.info("准备药学知识蒸馏数据...")
        
        for item in tqdm(knowledge_data, desc="处理知识数据"):
            question = item.get('question', '')
            category = item.get('category', 'general')
            difficulty = item.get('difficulty', 'medium')
            
            # 获取教师模型的回答作为软标签
            teacher_response = self.teacher_model.generate_response(question)
            
            # 获取学生模型的初始回答
            student_response = self.student_model.generate_response(question)
            
            # 计算知识蒸馏信号
            distillation_signal = self._calculate_distillation_signal(
                teacher_response.text, student_response.text
            )
            
            distilled_item = {
                'question': question,
                'category': category,
                'difficulty': difficulty,
                'teacher_response': teacher_response.text,
                'teacher_confidence': teacher_response.confidence,
                'student_response': student_response.text,
                'student_confidence': student_response.confidence,
                'distillation_signal': distillation_signal,
                'target_response': teacher_response.text  # 以教师响应为目标
            }
            
            distilled_data.append(distilled_item)
        
        logger.info(f"准备了 {len(distilled_data)} 个药学知识样本用于蒸馏")
        return distilled_data
    
    def _calculate_distillation_signal(self, teacher_text: str, student_text: str) -> float:
        """计算知识蒸馏信号强度"""
        # 基于文本相似度和质量计算蒸馏信号
        teacher_words = set(teacher_text.lower().split())
        student_words = set(student_text.lower().split())
        
        # 计算词汇重叠度
        overlap = len(teacher_words.intersection(student_words))
        union = len(teacher_words.union(student_words))
        similarity = overlap / union if union > 0 else 0
        
        # 考虑文本长度比例
        length_ratio = min(len(student_text), len(teacher_text)) / max(len(student_text), len(teacher_text))
        
        # 综合蒸馏信号
        signal = (similarity + length_ratio) / 2
        return signal
    
    def distill_knowledge(self, train_data: List[Dict[str, Any]], 
                         val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        执行知识蒸馏训练
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            
        Returns:
            Dict[str, Any]: 训练结果摘要
        """
        logger.info(f"开始知识蒸馏训练，共 {len(train_data)} 个样本")
        
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # 训练阶段
            epoch_losses = self._train_epoch(train_data, epoch)
            
            # 验证阶段
            if val_data and epoch % self.config.eval_interval == 0:
                eval_metrics = self._evaluate(val_data)
                self.evaluation_history.append({
                    'epoch': epoch,
                    'metrics': eval_metrics
                })
                logger.info(f"验证指标: {eval_metrics}")
            
            # 保存检查点
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch)
            
            # 记录训练历史
            self.training_history.append({
                'epoch': epoch,
                'losses': epoch_losses
            })
        
        # 标记学生模型为已训练
        self.student_model.mark_as_trained()
        
        # 生成训练摘要
        summary = self._generate_training_summary()
        
        logger.info("知识蒸馏训练完成")
        return summary
    
    def _train_epoch(self, train_data: List[Dict[str, Any]], epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        total_losses = {'total_loss': 0, 'hard_loss': 0, 'soft_loss': 0}
        
        # 使用tqdm显示进度
        progress_bar = tqdm(train_data, desc=f"训练 Epoch {epoch + 1}")
        
        for batch_idx, item in enumerate(progress_bar):
            # 模拟logits（在实际实现中，这应该从模型中获得）
            teacher_logits = self._text_to_logits(item['teacher_response'])
            student_logits = self._text_to_logits(item['student_response'])
            
            # 创建目标标签（这里使用教师响应作为目标）
            targets = torch.argmax(teacher_logits, dim=1)
            
            # 计算蒸馏损失
            losses = self.loss_function(student_logits, teacher_logits, targets)
            
            # 更新总损失
            for key in total_losses:
                total_losses[key] += losses[key].item()
            
            # 模拟学习过程
            learning_loss = self.student_model.learn_from_teacher(
                item['teacher_response'],
                item['student_response'],
                item['target_response']
            )
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Learning': f"{learning_loss:.4f}"
            })
        
        # 计算平均损失
        num_samples = len(train_data)
        avg_losses = {key: total_losses[key] / num_samples for key in total_losses}
        
        return avg_losses
    
    def _text_to_logits(self, text: str) -> torch.Tensor:
        """将文本转换为模拟的logits（用于演示）"""
        # 在实际实现中，这应该从模型的输出层获得
        vocab_size = 1000  # 假设词汇表大小
        batch_size = 1
        
        # 基于文本内容生成模拟logits
        text_hash = sum(ord(c) for c in text)
        torch.manual_seed(text_hash % 10000)
        
        logits = torch.randn(batch_size, vocab_size)
        return logits
    
    def _evaluate(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估模型性能"""
        metrics = {
            'similarity_to_teacher': 0,
            'response_quality': 0,
            'confidence_improvement': 0
        }
        
        for item in val_data:
            # 获取学生模型的回答
            student_response = self.student_model.generate_response(item['question'])
            
            # 计算与教师模型的相似度
            teacher_text = item['teacher_response']
            student_text = student_response.text
            similarity = self._calculate_text_similarity(teacher_text, student_text)
            
            # 计算响应质量
            quality = self._calculate_response_quality(student_response)
            
            # 计算置信度提升
            initial_confidence = item.get('student_confidence', 0)
            current_confidence = student_response.confidence
            confidence_improvement = current_confidence - initial_confidence
            
            metrics['similarity_to_teacher'] += similarity
            metrics['response_quality'] += quality
            metrics['confidence_improvement'] += confidence_improvement
        
        # 计算平均值
        num_samples = len(val_data)
        for key in metrics:
            metrics[key] /= num_samples
        
        return metrics
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def _calculate_response_quality(self, response: StudentResponse) -> float:
        """计算响应质量分数"""
        quality_score = 0
        
        # 基于长度
        if len(response.text) > 50:
            quality_score += 0.3
        
        # 基于置信度
        quality_score += response.confidence * 0.5
        
        # 基于是否包含关键信息
        key_indicators = ['药物', '治疗', '剂量', '注意', '建议']
        has_key_info = any(indicator in response.text for indicator in key_indicators)
        if has_key_info:
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _save_checkpoint(self, epoch: int):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'student_model_info': self.student_model.get_model_info()
        }
        
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.json')
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _generate_training_summary(self) -> Dict[str, Any]:
        """生成训练摘要"""
        if not self.training_history:
            return {'message': '无训练记录'}
        
        total_epochs = len(self.training_history)
        avg_total_loss = np.mean([h['losses']['total_loss'] for h in self.training_history])
        avg_hard_loss = np.mean([h['losses']['hard_loss'] for h in self.training_history])
        avg_soft_loss = np.mean([h['losses']['soft_loss'] for h in self.training_history])
        
        summary = {
            'total_epochs': total_epochs,
            'average_total_loss': avg_total_loss,
            'average_hard_loss': avg_hard_loss,
            'average_soft_loss': avg_soft_loss,
            'final_epoch': self.training_history[-1]['epoch'],
            'training_completed': True,
            'student_model_trained': self.student_model.is_trained,
            'output_dir': self.output_dir
        }
        
        # 保存训练摘要
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary
    
    def get_distillation_report(self) -> Dict[str, Any]:
        """获取知识蒸馏报告"""
        return {
            'config': self.config.to_dict(),
            'training_summary': self._generate_training_summary(),
            'teacher_model': self.teacher_model.get_model_info(),
            'student_model': self.student_model.get_model_info(),
            'total_training_samples': len(self.training_history),
            'output_directory': self.output_dir
        }