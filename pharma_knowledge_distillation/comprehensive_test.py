#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试Web界面响应完整性
"""
import sys
import os
import requests
import json
import time

# 添加项目路径
sys.path.append('e:\\data\\yangsitao_pharma_knowledge_distillation')

def comprehensive_test():
    """综合测试Web界面响应完整性"""
    
    print("=== 综合测试Web界面响应完整性 ===\n")
    
    # 测试1: 直接模型调用
    print("1. 直接模型调用测试...")
    try:
        from src.web.app import PharmaDistillationApp
        app = PharmaDistillationApp()
        app.initialize_models()
        
        question = "阿司匹林的副作用有哪些？"
        teacher_response = app.teacher_model.generate_response(question)
        
        print(f"   ✅ 教师模型响应长度: {len(teacher_response.text)} 字符")
        
        # 检查关键词
        keywords = ["胃肠道反应", "出血风险", "胃溃疡", "恶心"]
        found_keywords = [kw for kw in keywords if kw in teacher_response.text]
        print(f"   ✅ 找到关键词: {', '.join(found_keywords)}")
        
        if len(found_keywords) == len(keywords):
            print("   ✅ 所有关键词都找到")
        else:
            missing = set(keywords) - set(found_keywords)
            print(f"   ⚠️  缺少关键词: {', '.join(missing)}")
            
    except Exception as e:
        print(f"   ❌ 直接模型测试失败: {e}")
    
    # 测试2: Web界面方法调用
    print("\n2. Web界面方法调用测试...")
    try:
        teacher_text, student_text, evaluation = app.test_model_interaction(question)
        web_content = teacher_text.replace("🎓 教师模型:\n", "")
        
        print(f"   ✅ Web界面响应长度: {len(web_content)} 字符")
        
        # 检查关键词
        found_keywords_web = [kw for kw in keywords if kw in web_content]
        print(f"   ✅ Web界面找到关键词: {', '.join(found_keywords_web)}")
        
        if len(found_keywords_web) == len(keywords):
            print("   ✅ Web界面所有关键词都找到")
        else:
            missing_web = set(keywords) - set(found_keywords_web)
            print(f"   ⚠️  Web界面缺少关键词: {', '.join(missing_web)}")
            
        # 检查是否有截断迹象
        if "..." in web_content:
            print("   ⚠️  Web响应包含省略号，可能被截断")
        else:
            print("   ✅ Web响应不包含省略号")
            
    except Exception as e:
        print(f"   ❌ Web界面方法测试失败: {e}")
    
    # 测试3: Gradio API测试
    print("\n3. Gradio API测试...")
    try:
        # 使用Gradio的API
        api_url = "http://localhost:7860/gradio_api/call/test_model_interaction"
        
        payload = {
            "data": [question],
            "event_data": None,
            "fn_index": 0,
            "trigger_id": 0
        }
        
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API响应状态: {response.status_code}")
            print(f"   ✅ API响应格式: {type(result)}")
            
            if 'event_id' in result:
                # 需要获取结果
                event_id = result['event_id']
                time.sleep(2)  # 等待处理
                
                result_url = f"http://localhost:7860/gradio_api/call/result/{event_id}"
                result_response = requests.get(result_url, timeout=30)
                
                if result_response.status_code == 200:
                    final_result = result_response.json()
                    print(f"   ✅ 最终结果获取成功")
                    
                    if 'data' in final_result and len(final_result['data']) >= 2:
                        api_teacher = final_result['data'][0]
                        api_content = api_teacher.replace("🎓 教师模型:\n", "")
                        
                        print(f"   ✅ API教师响应长度: {len(api_content)} 字符")
                        
                        # 检查关键词
                        found_keywords_api = [kw for kw in keywords if kw in api_content]
                        print(f"   ✅ API找到关键词: {', '.join(found_keywords_api)}")
                        
                        if len(found_keywords_api) == len(keywords):
                            print("   ✅ API所有关键词都找到")
                        else:
                            missing_api = set(keywords) - set(found_keywords_api)
                            print(f"   ⚠️  API缺少关键词: {', '.join(missing_api)}")
                    else:
                        print(f"   ⚠️  API数据格式异常: {final_result}")
                else:
                    print(f"   ⚠️  无法获取最终结果: {result_response.status_code}")
            else:
                print(f"   ⚠️  意外的API响应格式: {result}")
        else:
            print(f"   ❌ API调用失败: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ API测试失败: {e}")
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print("建议:")
    print("1. 如果直接模型调用完整，但Web界面显示不完整，可能是显示问题")
    print("2. 检查浏览器控制台是否有错误信息")
    print("3. 考虑增加Gradio文本框的max_lines参数")
    print("4. 检查是否有CSS样式限制了显示高度")
    
    return True

if __name__ == "__main__":
    comprehensive_test()