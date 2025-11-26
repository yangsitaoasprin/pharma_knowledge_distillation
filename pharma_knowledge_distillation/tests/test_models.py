"""
模型测试
"""

import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.teacher_model import TeacherModel, TeacherResponse
from src.models.student_model import StudentModel, StudentResponse
from src.models.distillation import KnowledgeDistillationEngine, DistillationConfig

class TestModels:
    """模型测试类"""
    
    def test_teacher_model_creation(self):
        """测试教师模型创建"""
        # 创建教师模型（模拟）
        teacher = TeacherModel(model_name="test-teacher")
        
        # 验证模型信息
        info = teacher.get_model_info()
        assert info['type'] == 'teacher_model'
        assert info['name'] == 'test-teacher'
    
    def test_student_model_creation(self):
        """测试学生模型创建"""
        # 创建学生模型
        student = StudentModel(model_name="test-student")
        
        # 验证模型信息
        info = student.get_model_info()
        assert info['type'] == 'student_model'
        assert info['name'] == 'test-student'
        assert not info['is_trained']
    
    def test_teacher_response(self):
        """测试教师模型响应"""
        response = TeacherResponse(
            text="这是一个测试回答",
            confidence=0.85,
            metadata={'test': True}
        )
        
        assert response.text == "这是一个测试回答"
        assert response.confidence == 0.85
        assert response.metadata['test'] is True
    
    def test_student_response(self):
        """测试学生模型响应"""
        response = StudentResponse(
            text="这是一个学生回答",
            confidence=0.75,
            loss=0.25,
            metadata={'test': True}
        )
        
        assert response.text == "这是一个学生回答"
        assert response.confidence == 0.75
        assert response.loss == 0.25
        assert response.metadata['test'] is True
    
    def test_distillation_config(self):
        """测试蒸馏配置"""
        config = DistillationConfig(
            temperature=4.0,
            alpha=0.6,
            beta=0.4,
            epochs=5
        )
        
        assert config.temperature == 4.0
        assert config.alpha == 0.6
        assert config.beta == 0.4
        assert config.epochs == 5
    
    def test_student_model_learning(self):
        """测试学生模型学习功能"""
        student = StudentModel(model_name="test-student")
        
        # 模拟学习过程
        loss = student.learn_from_teacher(
            teacher_response="教师回答",
            student_response="学生回答", 
            target_response="目标回答"
        )
        
        assert isinstance(loss, float)
        assert 0.0 <= loss <= 2.0
        
        # 检查训练历史
        summary = student.get_training_summary()
        assert 'total_sessions' in summary
        assert summary['total_sessions'] == 1
    
    def test_distillation_engine_creation(self):
        """测试蒸馏引擎创建"""
        teacher = TeacherModel(model_name="test-teacher")
        student = StudentModel(model_name="test-student")
        config = DistillationConfig()
        
        engine = KnowledgeDistillationEngine(teacher, student, config)
        
        # 验证引擎创建
        assert engine.teacher_model == teacher
        assert engine.student_model == student
        assert engine.config == config

if __name__ == "__main__":
    pytest.main(["-v", __file__])