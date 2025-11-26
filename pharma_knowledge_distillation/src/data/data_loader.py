"""
数据加载器
提供高效的数据加载和批处理功能
"""

import json
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import random
from datetime import datetime

from .dataset import PharmaKnowledgeDataset
from .preprocessor import PharmaDataPreprocessor

logger = logging.getLogger(__name__)

class PharmaDataLoader:
    """药学数据加载器"""
    
    def __init__(self, dataset: PharmaKnowledgeDataset, batch_size: int = 4, 
                 shuffle: bool = True, num_workers: int = 0):
        """
        初始化数据加载器
        
        Args:
            dataset: 数据集实例
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # 创建PyTorch DataLoader
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"数据加载器初始化完成: batch_size={batch_size}, shuffle={shuffle}")
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        自定义批次处理函数
        
        Args:
            batch: 批次数据列表
            
        Returns:
            Dict[str, Any]: 处理后的批次数据
        """
        # 提取批次中的各个字段
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # 收集元数据
        metadata = [item['metadata'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
            'metadata': metadata,
            'batch_size': len(batch)
        }
    
    def __iter__(self):
        """迭代器"""
        return iter(self.data_loader)
    
    def __len__(self) -> int:
        """返回批次数量"""
        return len(self.data_loader)
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """获取批次统计信息"""
        total_batches = len(self.data_loader)
        total_samples = len(self.dataset)
        
        return {
            'total_samples': total_samples,
            'batch_size': self.batch_size,
            'total_batches': total_batches,
            'last_batch_size': total_samples % self.batch_size if total_samples % self.batch_size != 0 else self.batch_size
        }

class BalancedDataLoader:
    """平衡数据加载器 - 确保各类别样本平衡"""
    
    def __init__(self, dataset: PharmaKnowledgeDataset, batch_size: int = 4, 
                 samples_per_category: Optional[int] = None):
        """
        初始化平衡数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            samples_per_category: 每个类别的样本数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_category = samples_per_category
        
        # 按类别组织样本
        self.category_samples = self._organize_by_category()
        
        # 创建平衡采样器
        self.balanced_indices = self._create_balanced_indices()
        
        logger.info(f"平衡数据加载器初始化完成: {len(self.category_samples)} 个类别")
    
    def _organize_by_category(self) -> Dict[str, List[int]]:
        """按类别组织样本索引"""
        category_samples = defaultdict(list)
        
        for idx, sample in enumerate(self.dataset.samples):
            category = sample.get('category', 'unknown')
            category_samples[category].append(idx)
        
        return dict(category_samples)
    
    def _create_balanced_indices(self) -> List[int]:
        """创建平衡的样本索引列表"""
        balanced_indices = []
        
        # 确定每个类别的样本数
        if self.samples_per_category:
            samples_per_cat = self.samples_per_category
        else:
            # 使用最小类别的样本数
            samples_per_cat = min(len(indices) for indices in self.category_samples.values())
        
        # 为每个类别采样
        for category, indices in self.category_samples.items():
            if len(indices) >= samples_per_cat:
                # 随机采样
                sampled_indices = random.sample(indices, samples_per_cat)
            else:
                # 重复采样以达到目标数量
                sampled_indices = indices * (samples_per_cat // len(indices))
                remaining = samples_per_cat % len(indices)
                sampled_indices.extend(random.sample(indices, remaining))
            
            balanced_indices.extend(sampled_indices)
        
        # 打乱顺序
        random.shuffle(balanced_indices)
        
        return balanced_indices
    
    def __iter__(self) -> List[Dict[str, Any]]:
        """迭代器"""
        batch = []
        
        for idx in self.balanced_indices:
            sample = self.dataset[idx]
            batch.append(sample)
            
            if len(batch) == self.batch_size:
                yield self._collate_fn(batch)
                batch = []
        
        # 处理最后一个不完整的批次
        if batch:
            yield self._collate_fn(batch)
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批次处理函数"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'metadata': [item['metadata'] for item in batch],
            'batch_size': len(batch)
        }
    
    def __len__(self) -> int:
        """返回批次数量"""
        return len(self.balanced_indices) // self.batch_size + (1 if len(self.balanced_indices) % self.batch_size != 0 else 0)

class MultiTaskDataLoader:
    """多任务数据加载器 - 同时处理不同类型的药学问题"""
    
    def __init__(self, dataset: PharmaKnowledgeDataset, batch_size: int = 4):
        """
        初始化多任务数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 按任务类型组织数据
        self.task_data = self._organize_by_task()
        
        logger.info(f"多任务数据加载器初始化完成: {len(self.task_data)} 个任务类型")
    
    def _organize_by_task(self) -> Dict[str, List[Dict[str, Any]]]:
        """按任务类型组织数据"""
        task_data = defaultdict(list)
        
        for sample in self.dataset.samples:
            question_type = sample.get('question_type', 'general')
            task_data[question_type].append(sample)
        
        return dict(task_data)
    
    def get_task_batch(self, task_type: str) -> Optional[Dict[str, Any]]:
        """
        获取特定任务类型的批次
        
        Args:
            task_type: 任务类型
            
        Returns:
            Optional[Dict[str, Any]]: 批次数据
        """
        if task_type not in self.task_data:
            return None
        
        samples = self.task_data[task_type]
        if len(samples) < self.batch_size:
            return None
        
        # 随机采样一个批次
        batch_samples = random.sample(samples, self.batch_size)
        
        # 转换为模型输入格式
        batch = []
        for sample in batch_samples:
            model_input = {
                'input_ids': torch.tensor([hash(sample['question']) % 10000]),
                'attention_mask': torch.ones(1),
                'labels': torch.tensor([sample.get('category_id', 0)]),
                'metadata': sample
            }
            batch.append(model_input)
        
        return self._collate_fn(batch)
    
    def get_mixed_batch(self) -> Dict[str, Any]:
        """
        获取混合任务批次
        
        Returns:
            Dict[str, Any]: 混合批次数据
        """
        batch = []
        tasks_per_batch = min(len(self.task_data), self.batch_size)
        samples_per_task = self.batch_size // tasks_per_batch
        
        for task_type, samples in self.task_data.items():
            if len(batch) >= self.batch_size:
                break
            
            # 为该任务采样
            task_samples = random.sample(samples, min(samples_per_task, len(samples)))
            
            for sample in task_samples:
                if len(batch) >= self.batch_size:
                    break
                
                model_input = {
                    'input_ids': torch.tensor([hash(sample['question']) % 10000]),
                    'attention_mask': torch.ones(1),
                    'labels': torch.tensor([sample.get('category_id', 0)]),
                    'metadata': sample
                }
                batch.append(model_input)
        
        return self._collate_fn(batch)
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批次处理函数"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'metadata': [item['metadata'] for item in batch],
            'batch_size': len(batch),
            'task_types': [item['metadata'].get('question_type', 'general') for item in batch]
        }
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        return {
            'total_tasks': len(self.task_data),
            'task_distribution': {task: len(samples) for task, samples in self.task_data.items()},
            'batch_size': self.batch_size,
            'samples_per_task': {task: len(samples) // self.batch_size for task, samples in self.task_data.items()}
        }

class DataLoaderFactory:
    """数据加载器工厂类"""
    
    @staticmethod
    def create_dataloader(dataset: PharmaKnowledgeDataset, 
                         dataloader_type: str = 'standard',
                         **kwargs) -> Any:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            dataloader_type: 加载器类型 ('standard', 'balanced', 'multitask')
            **kwargs: 其他参数
            
        Returns:
            Any: 数据加载器实例
        """
        if dataloader_type == 'standard':
            return PharmaDataLoader(dataset, **kwargs)
        elif dataloader_type == 'balanced':
            return BalancedDataLoader(dataset, **kwargs)
        elif dataloader_type == 'multitask':
            return MultiTaskDataLoader(dataset, **kwargs)
        else:
            raise ValueError(f"不支持的数据加载器类型: {dataloader_type}")
    
    @staticmethod
    def create_from_file(data_path: str, 
                        dataloader_type: str = 'standard',
                        **kwargs) -> Any:
        """
        从文件创建数据加载器
        
        Args:
            data_path: 数据文件路径
            dataloader_type: 加载器类型
            **kwargs: 其他参数
            
        Returns:
            Any: 数据加载器实例
        """
        # 创建数据集
        dataset = PharmaKnowledgeDataset(data_path=data_path)
        
        # 创建数据加载器
        return DataLoaderFactory.create_dataloader(dataset, dataloader_type, **kwargs)
    
    @staticmethod
    def create_train_val_loaders(dataset: PharmaKnowledgeDataset, 
                               val_ratio: float = 0.2,
                               **kwargs) -> Tuple[Any, Any]:
        """
        创建训练和验证数据加载器
        
        Args:
            dataset: 数据集
            val_ratio: 验证集比例
            **kwargs: 其他参数
            
        Returns:
            Tuple[Any, Any]: (训练加载器, 验证加载器)
        """
        # 分割数据集
        train_dataset, val_dataset, _ = dataset.split_dataset(
            train_ratio=1-val_ratio, 
            val_ratio=val_ratio
        )
        
        # 创建数据加载器
        train_loader = DataLoaderFactory.create_dataloader(train_dataset, **kwargs)
        val_loader = DataLoaderFactory.create_dataloader(val_dataset, **kwargs)
        
        return train_loader, val_loader