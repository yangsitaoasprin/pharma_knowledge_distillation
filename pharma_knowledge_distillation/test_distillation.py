import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.evaluator import ModelEvaluator
from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
import time

# 初始化模型
teacher = TeacherModel('deepseek-r1:latest')
student = StudentModel('qwen2:0.5b')
evaluator = ModelEvaluator(teacher, student)

# 测试问题
test_questions = [
    '阿司匹林是什么药物？',
    '维生素D缺乏如何补充？',
    '氯化钾的作用是什么？'
]

print('🧪 知识蒸馏测试开始...')
print('='*60)

for question in test_questions:
    print(f'\n❓ 问题: {question}')
    
    # 获取教师回答
    teacher_response = teacher.generate_response(question)
    print(f'👨‍🏫 教师回答: {teacher_response.text[:100]}...')
    
    # 获取学生回答  
    student_response = student.generate_response(question)
    print(f'👨‍🎓 学生回答: {student_response.text[:100]}...')
    
    # 评估相似度
    result = evaluator.evaluate_single_response(question, teacher_response.text)
    
    print(f'📊 相似度: {result["metrics"]["similarity_to_teacher"]:.3f}')
    print(f'⭐ 质量分: {result["metrics"]["response_quality"]:.3f}')
    print(f'🎯 学生置信度: {result["metrics"]["student_confidence"]:.3f}')
    print('-'*50)
    
    time.sleep(1)  # 避免过快请求

print('\n✅ 测试完成！')