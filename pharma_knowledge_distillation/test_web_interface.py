#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Web界面的实际响应数据
"""
import sys
import os
import time

# 添加项目路径
sys.path.append('e:\\data\\yangsitao_pharma_knowledge_distillation')

# 导入Web应用
from src.web.app import PharmaDistillationApp

def test_web_interface():
    """测试Web界面响应"""
    try:
        # 创建应用实例
        app = PharmaDistillationApp()
        
        # 初始化模型
        print("正在初始化模型...")
        init_result = app.initialize_models()
        print(f"初始化结果: {init_result}")
        
        # 测试问题
        question = "阿司匹林的副作用有哪些？"
        
        print(f"\n=== 测试问题 ===")
        print(f"问题: {question}")
        
        # 调用测试方法
        print("正在调用test_model_interaction...")
        teacher_text, student_text, evaluation = app.test_model_interaction(question)
        
        print(f"\n=== Web界面教师响应 ===")
        print(f"响应长度: {len(teacher_text)} 字符")
        print(f"完整响应:\n{teacher_text}")
        
        print(f"\n=== Web界面学生响应 ===")
        print(f"响应长度: {len(student_text)} 字符")
        print(f"完整响应:\n{student_text}")
        
        # 检查完整性
        print(f"\n=== Web界面完整性检查 ===")
        keywords = ["胃肠道反应", "出血风险", "胃溃疡", "恶心"]
        
        for keyword in keywords:
            if keyword in teacher_text:
                print(f"✅ Web教师响应包含'{keyword}'")
            else:
                print(f"❌ Web教师响应缺少'{keyword}'")
                
        return True
        
    except Exception as e:
        print(f"Web界面测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_web_interface()