"""
训练模块测试
"""

import pytest
import sys
import os
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.loss_functions import DistillationLoss, LossFunctionFactory
from src.training.evaluator import ModelEvaluator
from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel

class TestTrainingModule:
    """训练模块测试类"""
    
    def test_distillation_loss(self):
        """测试蒸馏损失函数"""
        loss_fn = DistillationLoss(temperature=3.0, alpha=0.7, beta=0.3)
        
        # 创建模拟数据
        batch_size = 4
        vocab_size = 1000
        num_classes = 10
        
        student_logits = torch.randn(batch_size, vocab_size)
        teacher_logits = torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # 计算损失
        losses = loss_fn(student_logits, teacher_logits, targets)
        
        # 验证损失结果
        assert 'total_loss' in losses
        assert 'hard_loss' in losses
        assert 'soft_loss' in losses
        assert losses['total_loss'].item() > 0
        assert losses['hard_loss'].item() > 0
        assert losses['soft_loss'].item() > 0
    
    def test_loss_function_factory(self):
        """测试损失函数工厂"""
        # 测试创建不同类型的损失函数
        loss_types = LossFunctionFactory.get_available_loss_functions()
        
        for loss_type in loss_types:
            loss_fn = LossFunctionFactory.create_loss_function(loss_type)
            assert loss_fn is not None
    
    def test_evaluator_creation(self):
        """测试评估器创建"""
        # 创建模型
        teacher = TeacherModel(model_name="test-teacher")
        student = StudentModel(model_name="test-student")
        
        # 创建评估器
        evaluator = ModelEvaluator(teacher, student)
        
        # 验证评估器创建
        assert evaluator.teacher_model == teacher
        assert evaluator.student_model == student
    
    def test_similarity_calculation(self):
        """测试相似度计算"""
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        # 测试相似度计算
        text1 = "阿司匹林用于镇痛和抗炎"
        text2 = "阿司匹林具有镇痛抗炎作用"
        
        similarity = evaluator._calculate_similarity(text1, text2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.3  # 应该有一定的相似度
    
    def test_response_quality(self):
        """测试响应质量评估"""
        from src.models.student_model import StudentResponse
        
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        # 创建测试响应
        good_response = StudentResponse(
            text="阿司匹林是一种常用的解热镇痛药，主要通过抑制前列腺素合成发挥作用。常见副作用包括胃肠道不适。",
            confidence=0.85
        )
        
        bad_response = StudentResponse(
            text="我不知道，不清楚。",
            confidence=0.1
        )
        
        # 评估质量
        good_quality = evaluator._calculate_response_quality(good_response)
        bad_quality = evaluator._calculate_response_quality(bad_response)
        
        assert good_quality > bad_quality
        assert 0.0 <= good_quality <= 1.0
        assert 0.0 <= bad_quality <= 1.0
    
    def test_completeness_calculation(self):
        """测试完整性计算"""
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        teacher_response = "阿司匹林用于镇痛抗炎，常见副作用包括胃肠道反应和出血风险。建议饭后服用。"
        student_response = "阿司匹林用于镇痛，有胃肠道副作用。"
        
        completeness = evaluator._calculate_completeness(teacher_response, student_response)
        
        assert isinstance(completeness, float)
        assert 0.0 <= completeness <= 1.0
        assert completeness < 1.0  # 应该是不完整的
    
    def test_keyword_coverage(self):
        """测试关键词覆盖率"""
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        teacher_response = "阿司匹林具有镇痛、抗炎、解热作用，但可能引起胃肠道出血。"
        student_response = "阿司匹林可以镇痛，但可能有副作用。"
        
        coverage = evaluator._calculate_keyword_coverage(teacher_response, student_response)
        
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
        assert coverage > 0  # 应该有一些关键词覆盖
    
    def test_medical_accuracy_check(self):
        """测试医学准确性检查"""
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        # 测试矛盾术语
        contradictory_text = "这个药物绝对安全，但也很危险。"
        accuracy = evaluator._check_medical_accuracy(contradictory_text)
        
        assert accuracy['has_contradictory_terms'] is True
        assert len(accuracy['safety_warnings']) > 0
        
        # 测试不确定性术语
        uncertain_text = "这个药可能会有效果，我觉得应该有用。"
        accuracy = evaluator._check_medical_accuracy(uncertain_text)
        
        assert accuracy['has_uncertain_terms'] is True
    
    def test_question_type_classification(self):
        """测试问题类型分类"""
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        test_cases = [
            ("什么是药物？", "definition"),
            ("这个药怎么用？", "usage"),
            ("有什么副作用？", "side_effect"),
            ("能一起吃吗？", "interaction")
        ]
        
        for question, expected_type in test_cases:
            q_type = evaluator._classify_question_type(question)
            assert q_type == expected_type
    
    def test_improvement_suggestions(self):
        """测试改进建议生成"""
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        # 测试过短的回答
        short_response = "这个药有用。"
        suggestions = evaluator._generate_improvement_suggestions(short_response)
        
        assert len(suggestions) > 0
        assert any("过于简短" in s for s in suggestions)
        
        # 测试缺少专业术语的回答
        vague_response = "这个药物很好，应该按照说明使用。"
        suggestions = evaluator._generate_improvement_suggestions(vague_response)
        
        assert any("专业术语" in s for s in suggestions)
    
    def test_overall_score_calculation(self):
        """测试综合评分计算"""
        evaluator = ModelEvaluator(
            TeacherModel(model_name="test-teacher"),
            StudentModel(model_name="test-student")
        )
        
        metrics = {
            'similarity_to_teacher': 0.8,
            'response_quality': 0.9,
            'confidence_improvement': 0.3,
            'response_completeness': 0.7
        }
        
        overall_score = evaluator._calculate_overall_score(metrics)
        
        assert isinstance(overall_score, float)
        assert 0.0 <= overall_score <= 1.0
        assert overall_score > 0.5  # 应该有一个不错的分数

if __name__ == "__main__":
    pytest.main(["-v", __file__])