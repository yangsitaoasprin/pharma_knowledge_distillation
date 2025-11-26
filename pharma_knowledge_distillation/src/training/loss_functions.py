"""
损失函数模块
实现知识蒸馏的各种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7, beta: float = 0.3):
        """
        初始化知识蒸馏损失函数
        
        Args:
            temperature: 蒸馏温度
            alpha: 硬标签损失权重
            beta: 软标签损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        logger.info(f"蒸馏损失函数初始化: T={temperature}, α={alpha}, β={beta}")
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型logits [batch_size, vocab_size]
            teacher_logits: 教师模型logits [batch_size, vocab_size]
            targets: 真实标签 [batch_size]
            
        Returns:
            Dict[str, torch.Tensor]: 各种损失组件
        """
        # 确保所有张量都在同一设备上
        device = student_logits.device
        if teacher_logits.device != device:
            teacher_logits = teacher_logits.to(device)
        if targets.device != device:
            targets = targets.to(device)
            
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

class AttentionDistillationLoss(nn.Module):
    """注意力蒸馏损失函数"""
    
    def __init__(self, temperature: float = 1.0, gamma: float = 0.1):
        """
        初始化注意力蒸馏损失
        
        Args:
            temperature: 温度参数
            gamma: 注意力损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_attention: torch.Tensor, teacher_attention: torch.Tensor) -> torch.Tensor:
        """
        计算注意力蒸馏损失
        
        Args:
            student_attention: 学生模型注意力权重
            teacher_attention: 教师模型注意力权重
            
        Returns:
            torch.Tensor: 注意力损失
        """
        # 确保所有张量都在同一设备上
        device = student_attention.device
        if teacher_attention.device != device:
            teacher_attention = teacher_attention.to(device)
            
        # 确保维度匹配
        if student_attention.shape != teacher_attention.shape:
            # 使用插值调整大小
            student_attention = F.interpolate(
                student_attention.unsqueeze(1), 
                size=teacher_attention.shape[-2:], 
                mode='bilinear'
            ).squeeze(1)
        
        # 计算均方误差损失
        attention_loss = self.mse_loss(student_attention, teacher_attention)
        
        return self.gamma * attention_loss

class FeatureDistillationLoss(nn.Module):
    """特征蒸馏损失函数"""
    
    def __init__(self, temperature: float = 4.0, delta: float = 0.1):
        """
        初始化特征蒸馏损失
        
        Args:
            temperature: 温度参数
            delta: 特征损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.delta = delta
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征蒸馏损失
        
        Args:
            student_features: 学生模型特征
            teacher_features: 教师模型特征
            
        Returns:
            torch.Tensor: 特征损失
        """
        # 确保所有张量都在同一设备上
        device = student_features.device
        if teacher_features.device != device:
            teacher_features = teacher_features.to(device)
            
        # 特征维度对齐
        if student_features.shape != teacher_features.shape:
            # 使用线性变换对齐特征维度
            student_features = self._align_features(student_features, teacher_features.shape[-1])
        
        # 计算特征相似度损失
        feature_loss = self.mse_loss(student_features, teacher_features)
        
        return self.delta * feature_loss
    
    def _align_features(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        """对齐特征维度"""
        current_dim = features.shape[-1]
        device = features.device
        
        if current_dim < target_dim:
            # 填充零
            padding_size = target_dim - current_dim
            padding = torch.zeros(*features.shape[:-1], padding_size, device=device)
            return torch.cat([features, padding], dim=-1)
        elif current_dim > target_dim:
            # 截断
            return features[..., :target_dim]
        else:
            return features

class ContrastiveDistillationLoss(nn.Module):
    """对比蒸馏损失函数"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        """
        初始化对比蒸馏损失
        
        Args:
            temperature: 温度参数
            margin: 对比损失边界
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor,
                positive_pairs: torch.Tensor, negative_pairs: torch.Tensor) -> torch.Tensor:
        """
        计算对比蒸馏损失
        
        Args:
            student_features: 学生模型特征
            teacher_features: 教师模型特征
            positive_pairs: 正样本对
            negative_pairs: 负样本对
            
        Returns:
            torch.Tensor: 对比损失
        """
        # 确保所有张量都在同一设备上
        device = student_features.device
        if teacher_features.device != device:
            teacher_features = teacher_features.to(device)
        if positive_pairs.device != device:
            positive_pairs = positive_pairs.to(device)
        if negative_pairs.device != device:
            negative_pairs = negative_pairs.to(device)
            
        # 计算特征相似度
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        
        # 正样本对的相似度
        positive_sim = torch.sum(student_norm * teacher_norm, dim=1)
        
        # 负样本对的相似度
        negative_sim = torch.sum(student_norm * negative_pairs, dim=1)
        
        # 对比损失
        contrastive_loss = torch.relu(self.margin - positive_sim + negative_sim)
        
        return torch.mean(contrastive_loss)

class MultiTaskDistillationLoss(nn.Module):
    """多任务知识蒸馏损失函数"""
    
    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        """
        初始化多任务蒸馏损失
        
        Args:
            task_weights: 各任务权重
        """
        super().__init__()
        self.task_weights = task_weights or {
            'classification': 0.4,
            'generation': 0.4,
            'retrieval': 0.2
        }
        
        self.distillation_loss = DistillationLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_outputs: Dict[str, torch.Tensor], 
                teacher_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务蒸馏损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            targets: 目标标签
            
        Returns:
            Dict[str, torch.Tensor]: 各任务损失
        """
        total_loss = 0.0
        task_losses = {}
        
        for task_name, weight in self.task_weights.items():
            if task_name in student_outputs and task_name in teacher_outputs:
                if task_name == 'classification':
                    loss = self.ce_loss(student_outputs[task_name], targets[task_name])
                elif task_name == 'generation':
                    loss = self.distillation_loss(
                        student_outputs[task_name], 
                        teacher_outputs[task_name], 
                        targets[task_name]
                    )['total_loss']
                elif task_name == 'retrieval':
                    loss = self.mse_loss(student_outputs[task_name], teacher_outputs[task_name])
                else:
                    loss = self.mse_loss(student_outputs[task_name], teacher_outputs[task_name])
                
                task_losses[f'{task_name}_loss'] = loss
                total_loss += weight * loss
        
        task_losses['total_loss'] = total_loss
        return task_losses

class AdaptiveTemperatureLoss(nn.Module):
    """自适应温度蒸馏损失函数"""
    
    def __init__(self, initial_temperature: float = 3.0, adapt_rate: float = 0.1):
        """
        初始化自适应温度损失
        
        Args:
            initial_temperature: 初始温度
            adapt_rate: 自适应学习率
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.adapt_rate = adapt_rate
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算自适应温度蒸馏损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            
        Returns:
            Dict[str, torch.Tensor]: 损失和温度
        """
        # 自适应温度
        current_temp = torch.clamp(self.temperature, min=1.0, max=10.0)
        
        # 软标签损失
        soft_targets = F.softmax(teacher_logits / current_temp, dim=1)
        soft_predictions = F.log_softmax(student_logits / current_temp, dim=1)
        soft_loss = self.kl_loss(soft_predictions, soft_targets) * (current_temp ** 2)
        
        return {
            'soft_loss': soft_loss,
            'temperature': current_temp
        }

class CurriculumDistillationLoss(nn.Module):
    """课程学习蒸馏损失函数"""
    
    def __init__(self, difficulty_levels: int = 3, temperature: float = 3.0):
        """
        初始化课程蒸馏损失
        
        Args:
            difficulty_levels: 难度等级数
            temperature: 基础温度
        """
        super().__init__()
        self.difficulty_levels = difficulty_levels
        self.temperature = temperature
        self.current_difficulty = 0
        
        self.distillation_loss = DistillationLoss(temperature=temperature)
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, difficulty_levels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算课程蒸馏损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            targets: 目标标签
            difficulty_levels: 难度等级
            
        Returns:
            Dict[str, torch.Tensor]: 课程损失
        """
        total_loss = 0.0
        level_losses = {}
        
        for level in range(self.current_difficulty + 1):
            # 选择当前难度等级的样本
            level_mask = (difficulty_levels == level)
            
            if level_mask.sum() > 0:
                level_student_logits = student_logits[level_mask]
                level_teacher_logits = teacher_logits[level_mask]
                level_targets = targets[level_mask]
                
                # 计算该难度等级的损失
                level_loss = self.distillation_loss(
                    level_student_logits, 
                    level_teacher_logits, 
                    level_targets
                )
                
                level_losses[f'level_{level}_loss'] = level_loss['total_loss']
                total_loss += level_loss['total_loss']
        
        level_losses['total_loss'] = total_loss
        level_losses['current_difficulty'] = self.current_difficulty
        
        return level_losses
    
    def increase_difficulty(self):
        """增加难度等级"""
        if self.current_difficulty < self.difficulty_levels - 1:
            self.current_difficulty += 1
            logger.info(f"课程难度已增加到等级 {self.current_difficulty}")

class MetaDistillationLoss(nn.Module):
    """元学习蒸馏损失函数"""
    
    def __init__(self, meta_learning_rate: float = 0.01):
        """
        初始化元学习蒸馏损失
        
        Args:
            meta_learning_rate: 元学习率
        """
        super().__init__()
        self.meta_lr = meta_learning_rate
        self.distillation_loss = DistillationLoss()
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, support_set: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算元学习蒸馏损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            targets: 目标标签
            support_set: 支持集数据
            
        Returns:
            Dict[str, torch.Tensor]: 元学习损失
        """
        # 基础蒸馏损失
        base_loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        # 元学习损失（基于支持集）
        meta_loss = self._compute_meta_loss(support_set)
        
        # 组合损失
        total_loss = base_loss['total_loss'] + self.meta_lr * meta_loss
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss['total_loss'],
            'meta_loss': meta_loss
        }
    
    def _compute_meta_loss(self, support_set: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算元学习损失"""
        # 简化的元学习损失计算
        if 'student_logits' in support_set and 'teacher_logits' in support_set:
            student_logits = support_set['student_logits']
            teacher_logits = support_set['teacher_logits']
            targets = support_set.get('targets', torch.argmax(teacher_logits, dim=1))
            
            support_loss = self.distillation_loss(student_logits, teacher_logits, targets)
            return support_loss['total_loss']
        
        return torch.tensor(0.0)

class LossFunctionFactory:
    """损失函数工厂类"""
    
    @staticmethod
    def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
        """
        创建损失函数
        
        Args:
            loss_type: 损失函数类型
            **kwargs: 参数
            
        Returns:
            nn.Module: 损失函数实例
        """
        loss_functions = {
            'distillation': DistillationLoss,
            'attention': AttentionDistillationLoss,
            'feature': FeatureDistillationLoss,
            'contrastive': ContrastiveDistillationLoss,
            'multitask': MultiTaskDistillationLoss,
            'adaptive': AdaptiveTemperatureLoss,
            'curriculum': CurriculumDistillationLoss,
            'meta': MetaDistillationLoss
        }
        
        if loss_type not in loss_functions:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
        
        return loss_functions[loss_type](**kwargs)
    
    @staticmethod
    def get_available_loss_functions() -> List[str]:
        """获取可用的损失函数类型"""
        return [
            'distillation', 'attention', 'feature', 'contrastive',
            'multitask', 'adaptive', 'curriculum', 'meta'
        ]