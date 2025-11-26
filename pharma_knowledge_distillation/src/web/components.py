"""
Web界面组件
提供各种交互式组件和可视化工具
"""

import gradio as gr
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理组件"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.models = {}
        self.training_history = []
        self.evaluation_results = {}
    
    def register_model(self, model_name: str, model_info: Dict[str, Any]):
        """注册模型"""
        self.models[model_name] = model_info
        logger.info(f"模型已注册: {model_name}")
    
    def get_model_status(self) -> str:
        """获取模型状态"""
        status_text = "📊 模型状态概览\n\n"
        
        for model_name, info in self.models.items():
            status_text += f"🤖 {model_name}:\n"
            status_text += f"   类型: {info.get('type', 'unknown')}\n"
            status_text += f"   状态: {'已就绪' if info.get('ready', False) else '未就绪'}\n"
            status_text += f"   描述: {info.get('description', '无描述')}\n\n"
        
        return status_text
    
    def save_model_info(self, file_path: str):
        """保存模型信息"""
        model_info = {
            'models': self.models,
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.models)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型信息已保存: {file_path}")

class TrainingVisualizer:
    """训练可视化组件"""
    
    def __init__(self):
        """初始化训练可视化器"""
        self.training_data = []
        self.evaluation_data = []
    
    def add_training_data(self, epoch: int, metrics: Dict[str, float]):
        """添加训练数据"""
        data_point = {'epoch': epoch, **metrics}
        self.training_data.append(data_point)
    
    def add_evaluation_data(self, epoch: int, metrics: Dict[str, float]):
        """添加评估数据"""
        data_point = {'epoch': epoch, **metrics}
        self.evaluation_data.append(data_point)
    
    def generate_training_curves(self) -> go.Figure:
        """生成训练曲线图"""
        if not self.training_data:
            # 生成示例数据
            epochs = list(range(10))
            total_loss = [2.0 - 0.15 * i + 0.1 * np.random.randn() for i in epochs]
            learning_loss = [1.5 - 0.12 * i + 0.08 * np.random.randn() for i in epochs]
        else:
            epochs = [d['epoch'] for d in self.training_data]
            total_loss = [d.get('total_loss', 0) for d in self.training_data]
            learning_loss = [d.get('learning_loss', 0) for d in self.training_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs, y=total_loss,
            mode='lines+markers',
            name='总损失',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=learning_loss,
            mode='lines+markers',
            name='学习损失',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='训练损失曲线',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def generate_model_comparison(self) -> go.Figure:
        """生成模型对比图"""
        # 模拟对比数据
        categories = ['相似度', '质量', '置信度', '完整性', '响应速度']
        teacher_scores = [0.95, 0.92, 0.88, 0.90, 0.85]
        student_scores = [0.78, 0.82, 0.75, 0.80, 0.95]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='教师模型',
            x=categories,
            y=teacher_scores,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='学生模型',
            x=categories,
            y=student_scores,
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='教师模型 vs 学生模型性能对比',
            xaxis_title='评估维度',
            yaxis_title='得分',
            barmode='group',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def generate_response_analysis(self) -> go.Figure:
        """生成响应分析图"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('响应长度分布', '置信度分布', '响应时间趋势', '质量评分'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 响应长度分布
        response_lengths = np.random.normal(150, 50, 100)
        response_lengths = np.clip(response_lengths, 20, 500)
        
        fig.add_trace(
            go.Histogram(x=response_lengths, name='响应长度', marker_color='green'),
            row=1, col=1
        )
        
        # 置信度分布
        confidences = np.random.beta(2, 5, 100)
        fig.add_trace(
            go.Histogram(x=confidences, name='置信度', marker_color='purple'),
            row=1, col=2
        )
        
        # 响应时间趋势
        time_points = list(range(20))
        response_times = [1.0 + 0.1 * i + 0.2 * np.random.randn() for i in time_points]
        
        fig.add_trace(
            go.Scatter(x=time_points, y=response_times, mode='lines+markers',
                      name='响应时间', line=dict(color='red')),
            row=2, col=1
        )
        
        # 质量评分
        quality_metrics = ['准确性', '完整性', '专业性', '可读性']
        quality_scores = [0.85, 0.78, 0.82, 0.90]
        
        fig.add_trace(
            go.Bar(x=quality_metrics, y=quality_scores, name='质量评分',
                  marker_color=['blue', 'orange', 'green', 'red']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='响应分析综合图表',
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def generate_plot(self, plot_type: str) -> go.Figure:
        """生成指定类型的图表"""
        if plot_type == "训练曲线":
            return self.generate_training_curves()
        elif plot_type == "模型对比":
            return self.generate_model_comparison()
        elif plot_type == "响应分析":
            return self.generate_response_analysis()
        else:
            # 默认返回训练曲线
            return self.generate_training_curves()

class ResponseComparator:
    """响应对比分析组件"""
    
    def __init__(self):
        """初始化响应对比器"""
        self.comparison_history = []
    
    def compare_responses(self, question: str) -> str:
        """
        对比分析教师和学生模型的响应
        
        Args:
            question: 输入问题
            
        Returns:
            str: HTML格式的对比分析结果
        """
        if not question.strip():
            return "<p>请输入有效的问题</p>"
        
        # 模拟教师和学生模型的回答
        teacher_response = self._generate_teacher_response(question)
        student_response = self._generate_student_response(question)
        
        # 进行分析
        similarity = self._calculate_similarity(teacher_response, student_response)
        quality_analysis = self._analyze_response_quality(student_response)
        completeness_analysis = self._analyze_completeness(teacher_response, student_response)
        
        # 生成HTML报告
        html_report = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">
            <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
                🔍 响应对比分析报告
            </h2>
            
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #495057; margin-top: 0;">❓ 问题</h3>
                <p style="font-size: 16px; color: #2c3e50; font-weight: 500;">{question}</p>
            </div>
            
            <div style="display: flex; gap: 20px; margin: 20px 0;">
                <div style="flex: 1; background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h3 style="color: #2980b9; margin-top: 0;">🎓 教师模型回答</h3>
                    <p style="color: #2c3e50; font-size: 14px; line-height: 1.5;">{teacher_response}</p>
                </div>
                
                <div style="flex: 1; background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #f39c12;">
                    <h3 style="color: #d68910; margin-top: 0;">👨‍🎓 学生模型回答</h3>
                    <p style="color: #2c3e50; font-size: 14px; line-height: 1.5;">{student_response}</p>
                </div>
            </div>
            
            <div style="background-color: #f1f3f4; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #2c3e50; margin-top: 0;">📊 对比分析结果</h3>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                    <div style="background-color: white; padding: 10px; border-radius: 6px; text-align: center;">
                        <h4 style="color: #27ae60; margin: 5px 0;">相似度</h4>
                        <p style="font-size: 24px; font-weight: bold; color: #27ae60; margin: 5px 0;">{similarity:.3f}</p>
                    </div>
                    
                    <div style="background-color: white; padding: 10px; border-radius: 6px; text-align: center;">
                        <h4 style="color: #e74c3c; margin: 5px 0;">质量评分</h4>
                        <p style="font-size: 24px; font-weight: bold; color: #e74c3c; margin: 5px 0;">{quality_analysis['score']:.3f}</p>
                    </div>
                    
                    <div style="background-color: white; padding: 10px; border-radius: 6px; text-align: center;">
                        <h4 style="color: #8e44ad; margin: 5px 0;">完整性</h4>
                        <p style="font-size: 24px; font-weight: bold; color: #8e44ad; margin: 5px 0;">{completeness_analysis['score']:.3f}</p>
                    </div>
                </div>
            </div>
            
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #27ae60; margin-top: 0;">✅ 学生模型优势</h3>
                <ul style="color: #2c3e50; margin: 10px 0; padding-left: 20px;">
                    {''.join(f'<li>{adv}</li>' for adv in quality_analysis['advantages'])}
                </ul>
            </div>
            
            <div style="background-color: #fdf2e9; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #e67e22; margin-top: 0;">⚠️ 改进建议</h3>
                <ul style="color: #2c3e50; margin: 10px 0; padding-left: 20px;">
                    {''.join(f'<li>{suggestion}</li>' for suggestion in quality_analysis['suggestions'])}
                </ul>
            </div>
            
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #6c757d; margin-top: 0;">📈 详细分析</h3>
                <p style="color: #2c3e50; margin: 10px 0;"><strong>内容覆盖度:</strong> {completeness_analysis['coverage']:.1%}</p>
                <p style="color: #2c3e50; margin: 10px 0;"><strong>关键信息点:</strong> {completeness_analysis['key_points']}</p>
                <p style="color: #2c3e50; margin: 10px 0;"><strong>缺失内容:</strong> {completeness_analysis['missing_points']}</p>
            </div>
            
            <div style="text-align: center; margin-top: 20px; padding-top: 15px; border-top: 1px solid #dee2e6;">
                <p style="color: #6c757d; font-size: 12px;">
                    分析报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </div>
        """
        
        # 保存到历史记录
        self.comparison_history.append({
            'question': question,
            'teacher_response': teacher_response,
            'student_response': student_response,
            'similarity': similarity,
            'timestamp': datetime.now().isoformat()
        })
        
        return html_report
    
    def _generate_teacher_response(self, question: str) -> str:
        """生成教师模型回答（模拟）"""
        # 基于问题类型生成专业的回答
        if '副作用' in question:
            return ("阿司匹林的常见副作用包括：\n"
                   "1. 胃肠道反应：恶心、呕吐、胃痛、胃溃疡\n"
                   "2. 出血风险：可能增加出血倾向\n"
                   "3. 过敏反应：皮疹、哮喘样症状\n"
                   "4. 肾功能影响：长期使用可能影响肾功能\n\n"
                   "建议：饭后服用，避免空腹，如有不适及时就医。")
        
        elif '储存' in question or '保存' in question:
            return ("胰岛素的正确储存方法：\n"
                   "1. 未开封：2-8°C冷藏保存，避免冷冻\n"
                   "2. 已开封：室温保存（不超过25°C），4周内使用\n"
                   "3. 避免阳光直射和高温\n"
                   "4. 不要剧烈摇晃\n\n"
                   "注意：使用前检查有效期和药液状态。")
        
        elif '抗生素' in question:
            return ("抗生素使用的基本原则：\n"
                   "1. 合理用药：根据病原菌选择合适的抗生素\n"
                   "2. 足量足疗程：按医嘱完成整个疗程\n"
                   "3. 避免滥用：不用于病毒感染\n"
                   "4. 注意耐药性：避免不必要的使用\n\n"
                   "重要：必须在医生指导下使用，不可自行停药。")
        
        else:
            return ("这是一个专业的药学问题。\n\n"
                   "基于我的专业知识，我可以提供以下信息：\n"
                   "1. 药理作用机制\n"
                   "2. 临床应用指导\n"
                   "3. 安全性注意事项\n"
                   "4. 个体化用药建议\n\n"
                   "建议咨询专业医生或药师获取更详细的指导。")
    
    def _generate_student_response(self, question: str) -> str:
        """生成学生模型回答（模拟）"""
        # 生成相对简单但仍专业的回答
        if '副作用' in question:
            return ("根据我的学习，阿司匹林的主要副作用有：\n"
                   "- 胃肠道不适\n"
                   "- 出血风险增加\n"
                   "- 可能的过敏反应\n\n"
                   "建议饭后服用，如有严重不适应及时就医。我还在学习中，建议咨询专业医生。")
        
        elif '储存' in question or '保存' in question:
            return ("胰岛素应该这样储存：\n"
                   "- 没打开的放冰箱（2-8度）\n"
                   "- 打开了的室温保存\n"
                   "- 避免阳光直晒\n\n"
                   "这是我学到的知识，具体使用方法请咨询医生。")
        
        elif '抗生素' in question:
            return ("抗生素使用要注意：\n"
                   "- 按医生开的用\n"
                   "- 完成整个疗程\n"
                   "- 不要滥用\n\n"
                   "我还在学习更多药学知识，建议听从专业指导。")
        
        else:
            return ("根据我目前的学习，这是一个重要的药学问题。\n\n"
                   "我了解到的基本信息包括药理作用、用法用量等，\n"
                   "但我的知识还在积累中，建议咨询专业医生或药师\n"
                   "获取更准确和完整的指导。")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简化的相似度计算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def _analyze_response_quality(self, response: str) -> Dict[str, Any]:
        """分析响应质量"""
        # 质量评分（0-1）
        quality_score = 0.7 + 0.2 * np.random.random()  # 0.7-0.9之间
        
        # 优势分析
        advantages = [
            "回答简洁明了",
            "包含关键安全提醒",
            "语言通俗易懂",
            "结构清晰"
        ]
        
        # 改进建议
        suggestions = [
            "可以增加更多专业细节",
            "建议提供更多具体案例",
            "可以加强药理机制的解释",
            "建议增加个体化用药指导"
        ]
        
        return {
            'score': quality_score,
            'advantages': np.random.choice(advantages, size=2, replace=False).tolist(),
            'suggestions': np.random.choice(suggestions, size=2, replace=False).tolist()
        }
    
    def _analyze_completeness(self, teacher_response: str, student_response: str) -> Dict[str, Any]:
        """分析完整性"""
        # 模拟完整性分析
        coverage = 0.75 + 0.2 * np.random.random()  # 0.75-0.95之间
        
        return {
            'score': coverage,
            'coverage': coverage,
            'key_points': '包含主要药理信息',
            'missing_points': '缺少具体剂量指导'
        }