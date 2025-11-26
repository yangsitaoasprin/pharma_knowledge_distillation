#!/usr/bin/env python3
"""
评估功能修复验证脚本
"""

import sys
sys.path.append('src')

from web.app import PharmaDistillationApp
from models.teacher_model import TeacherModel
from models.student_model import StudentModel

def test_evaluation_functionality():
    """测试评估功能"""
    print("=== 开始评估功能测试 ===")
    
    try:
        # 初始化模型
        print("1. 初始化模型...")
        teacher = TeacherModel(model_name='deepseek-r1:latest')
        student = StudentModel(model_name='qwen2:0.5b')
        app = PharmaDistillationApp()
        
        # 设置模型
        app.teacher_model = teacher
        app.student_model = student
        
        if not (app.teacher_model and app.student_model):
            print("❌ 模型初始化失败")
            return False
            
        print("✅ 模型初始化成功")
        
        # 初始化评估器
        print("2. 初始化评估器...")
        from training.evaluator import ModelEvaluator
        app.evaluator = ModelEvaluator(app.teacher_model, app.student_model)
        print("✅ 评估器初始化成功")
        
        # 模拟训练完成状态
        print("3. 设置训练状态...")
        student.is_trained = True
        student.trained_epochs = 5
        student.best_metrics = {'loss': 0.1, 'similarity': 0.8}
        print("✅ 训练状态设置完成")
        
        # 测试评估报告生成
        print("4. 测试评估报告生成...")
        app.current_training_data = None  # 使用默认问题
        report = app.generate_evaluation_report()
        
        if "评估报告" in report and "核心指标" in report:
            print("✅ 评估报告生成成功")
            print(f"   报告长度: {len(report)} 字符")
        else:
            print("❌ 评估报告生成失败")
            return False
            
        # 测试批量评估
        print("5. 测试批量评估...")
        test_questions = [
            {'question': '阿司匹林的主要作用是什么？', 'teacher_response': ''},
            {'question': '如何正确储存胰岛素？', 'teacher_response': ''},
            {'question': '抗生素使用有哪些注意事项？', 'teacher_response': ''}
        ]
        
        batch_result = app.evaluator.evaluate_batch(test_questions)
        required_metrics = ['similarity_to_teacher', 'response_quality', 'confidence_improvement', 
                           'response_completeness', 'keyword_coverage', 'overall_score']
        
        if all(metric in batch_result for metric in required_metrics):
            print("✅ 批量评估成功")
            print("   评估指标:")
            for metric, value in batch_result.items():
                print(f"   - {metric}: {value:.4f}")
        else:
            print("❌ 批量评估失败")
            return False
            
        # 测试训练数据评估
        print("6. 测试训练数据评估...")
        training_data = {
            'train': [
                {'question': '什么是高血压？', 'teacher_response': '高血压是指血压持续升高的疾病。'},
                {'question': '糖尿病如何管理？', 'teacher_response': '糖尿病需要通过饮食、运动和药物综合管理。'}
            ],
            'validation': [
                {'question': '心脏病的症状有哪些？', 'teacher_response': '心脏病症状包括胸痛、呼吸困难等。'}
            ]
        }
        
        app.current_training_data = training_data
        validation_report = app.generate_evaluation_report()
        
        if "验证集评估报告" in validation_report:
            print("✅ 训练数据评估成功")
        else:
            print("❌ 训练数据评估失败")
            return False
            
        print("\n=== 所有测试通过！评估功能已完全修复 ===")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_evaluation_functionality()
    sys.exit(0 if success else 1)