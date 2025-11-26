"""
数据模块测试
"""

import pytest
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import PharmaKnowledgeDataset
from src.data.preprocessor import PharmaDataPreprocessor
from src.data.data_loader import PharmaDataLoader, BalancedDataLoader

class TestDataModule:
    """数据模块测试类"""
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        # 创建示例数据
        sample_data = [
            {
                "id": "test_001",
                "question": "测试问题1？",
                "category": "测试类别",
                "difficulty": "easy",
                "keywords": ["测试", "问题"]
            },
            {
                "id": "test_002", 
                "question": "测试问题2？",
                "category": "测试类别",
                "difficulty": "medium",
                "keywords": ["测试", "问题2"]
            }
        ]
        
        dataset = PharmaKnowledgeDataset(data_list=sample_data)
        
        # 验证数据集
        assert len(dataset) == 2
        assert dataset.split == 'train'
        
        # 验证样本
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert 'metadata' in sample
    
    def test_dataset_split(self):
        """测试数据集分割"""
        # 创建足够大的数据集用于分割测试
        sample_data = []
        for i in range(20):
            sample_data.append({
                "id": f"test_{i:03d}",
                "question": f"测试问题{i}？",
                "category": "测试类别",
                "difficulty": "easy",
                "keywords": ["测试"]
            })
        
        dataset = PharmaKnowledgeDataset(data_list=sample_data)
        train_dataset, val_dataset, test_dataset = dataset.split_dataset()
        
        # 验证分割结果
        assert len(train_dataset) == 14  # 70%
        assert len(val_dataset) == 4     # 20% 
        assert len(test_dataset) == 2    # 10%
    
    def test_preprocessor(self):
        """测试数据预处理器"""
        preprocessor = PharmaDataPreprocessor()
        
        # 测试文本清理
        dirty_text = "这是一个  测试文本！！！包含特殊字符@@@"
        clean_text = preprocessor.clean_text(dirty_text)
        
        assert "测试文本" in clean_text
        assert "@@@" not in clean_text
        
        # 测试医学实体提取
        medical_text = "阿司匹林和布洛芬都是常用药物"
        entities = preprocessor.extract_medical_entities(medical_text)
        
        assert isinstance(entities, dict)
        assert 'drugs' in entities
    
    def test_question_classification(self):
        """测试问题分类"""
        preprocessor = PharmaDataPreprocessor()
        
        test_cases = [
            ("什么是药物相互作用？", "definition"),
            ("这个药怎么用？", "usage"),
            ("有什么副作用？", "side_effect"),
            ("这两种药能一起吃吗？", "interaction")
        ]
        
        for question, expected_type in test_cases:
            q_type = preprocessor.classify_question_type(question)
            assert q_type == expected_type
    
    def test_data_augmentation(self):
        """测试数据增强"""
        preprocessor = PharmaDataPreprocessor()
        
        original_question = "什么是药物的副作用？"
        augmented = preprocessor.augment_question(original_question, 'paraphrase')
        
        assert len(augmented) > 1
        assert original_question in augmented
        assert any("请解释" in q for q in augmented)
    
    def test_data_validation(self):
        """测试数据验证"""
        preprocessor = PharmaDataPreprocessor()
        
        # 有效数据
        valid_data = {
            "question": "这是一个有效问题？",
            "category": "药物副作用",
            "difficulty": "easy"
        }
        
        is_valid, errors = preprocessor.validate_pharma_data(valid_data)
        assert is_valid is True
        assert len(errors) == 0
        
        # 无效数据
        invalid_data = {
            "question": "无效问题",  # 缺少问号
            "category": "无效类别",
            "difficulty": "invalid"
        }
        
        is_valid, errors = preprocessor.validate_pharma_data(invalid_data)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_data_loader(self):
        """测试数据加载器"""
        # 创建测试数据
        sample_data = []
        for i in range(10):
            sample_data.append({
                "id": f"test_{i:03d}",
                "question": f"测试问题{i}？",
                "category": "药物副作用",
                "difficulty": "easy",
                "keywords": ["测试"]
            })
        
        dataset = PharmaKnowledgeDataset(data_list=sample_data)
        data_loader = PharmaDataLoader(dataset, batch_size=4)
        
        # 验证数据加载器
        assert len(data_loader) == 3  # 10个样本，批次大小4，共3个批次
        
        # 测试迭代
        batch_count = 0
        for batch in data_loader:
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            assert 'metadata' in batch
            batch_count += 1
        
        assert batch_count == 3
    
    def test_balanced_data_loader(self):
        """测试平衡数据加载器"""
        # 创建多类别数据
        sample_data = []
        categories = ["药物副作用", "药物储存", "用药原则"]
        
        for i in range(30):
            sample_data.append({
                "id": f"test_{i:03d}",
                "question": f"测试问题{i}？",
                "category": categories[i % 3],
                "difficulty": "easy",
                "keywords": ["测试"]
            })
        
        dataset = PharmaKnowledgeDataset(data_list=sample_data)
        balanced_loader = BalancedDataLoader(dataset, batch_size=6, samples_per_category=4)
        
        # 验证平衡加载
        batch_count = 0
        for batch in balanced_loader:
            assert len(batch['metadata']) <= 6
            batch_count += 1
        
        assert batch_count > 0
    
    def test_dataset_statistics(self):
        """测试数据集统计"""
        # 创建多样化数据
        sample_data = [
            {
                "id": "test_001",
                "question": "短问题？",
                "category": "药物副作用",
                "difficulty": "easy",
                "keywords": ["短"]
            },
            {
                "id": "test_002",
                "question": "这是一个比较长的测试问题用于测试问题长度的计算？",
                "category": "药物储存",
                "difficulty": "medium",
                "keywords": ["长问题"]
            }
        ]
        
        dataset = PharmaKnowledgeDataset(data_list=sample_data)
        stats = dataset.get_dataset_statistics()
        
        assert stats['total_samples'] == 2
        assert len(stats['categories']) == 2
        assert 'easy' in stats['difficulty_distribution']
        assert 'medium' in stats['difficulty_distribution']
        assert stats['average_question_length'] > 0

if __name__ == "__main__":
    pytest.main(["-v", __file__])