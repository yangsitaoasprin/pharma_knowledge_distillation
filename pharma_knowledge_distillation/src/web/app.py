"""
Web应用主入口
使用Gradio创建交互式界面
"""

import gradio as gr
import logging
import sys
import os
from typing import Dict, Any, List, Tuple
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 添加项目根目录到Python路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.models.distillation import KnowledgeDistillationEngine, DistillationConfig
from src.data.dataset import PharmaKnowledgeDataset
from src.training.trainer import DistillationTrainer
from src.training.evaluator import ModelEvaluator
from src.web.components import ModelManager, TrainingVisualizer, ResponseComparator

logger = logging.getLogger(__name__)

class PharmaDistillationApp:
    """药学知识蒸馏Web应用"""
    
    def __init__(self):
        """初始化应用"""
        self.teacher_model = None
        self.student_model = None
        self.distillation_engine = None
        self.trainer = None
        self.evaluator = None
        self.current_training_data = None
        
        # 初始化模型管理器
        self.model_manager = ModelManager()
        
        logger.info("药学知识蒸馏Web应用初始化完成")
    
    def initialize_models(self, teacher_model_name: str = "deepseek-r1:latest", 
                         student_model_name: str = "qwen2:0.5b") -> str:
        """
        初始化模型
        
        Args:
            teacher_model_name: 教师模型名称
            student_model_name: 学生模型名称
            
        Returns:
            str: 初始化状态信息
        """
        try:
            # 初始化教师模型
            self.teacher_model = TeacherModel(model_name=teacher_model_name)
            
            # 初始化学生模型
            self.student_model = StudentModel(model_name=student_model_name)
            
            # 初始化评估器
            self.evaluator = ModelEvaluator(self.teacher_model, self.student_model)
            
            status = f"""
            ✅ 模型初始化成功！
            
            教师模型: {teacher_model_name}
            学生模型: {student_model_name}
            
            模型状态:
            - 教师模型已就绪
            - 学生模型已就绪
            - 评估器已初始化
            """
            
            logger.info(f"模型初始化成功: {teacher_model_name} -> {student_model_name}")
            return status
            
        except Exception as e:
            error_msg = f"❌ 模型初始化失败: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def test_model_interaction(self, question: str) -> Tuple[str, str, str]:
        """
        测试模型交互
        
        Args:
            question: 测试问题
            
        Returns:
            Tuple[str, str, str]: (教师回答, 学生回答, 评估结果)
        """
        if not self.teacher_model or not self.student_model:
            return "请先初始化模型", "请先初始化模型", "请先初始化模型"
        
        try:
            # 获取教师模型回答
            teacher_response = self.teacher_model.generate_response(question)
            
            # 获取学生模型回答
            student_response = self.student_model.generate_response(question)
            
            # 评估响应
            evaluation = self.evaluator.evaluate_single_response(question)
            
            # 格式化评估结果
            eval_text = f"""
            📊 评估结果:
            
            相似度: {evaluation['metrics']['similarity_to_teacher']:.3f}
            质量分: {evaluation['metrics']['response_quality']:.3f}
            完整性: {evaluation['metrics']['response_completeness']:.3f}
            学生置信度: {evaluation['metrics']['student_confidence']:.3f}
            
            🔍 详细分析:
            {evaluation['detailed_analysis']['medical_accuracy']}
            """
            
            return (f"🎓 教师模型:\n{teacher_response.text}",
                   f"👨‍🎓 学生模型:\n{student_response.text}",
                   eval_text)
            
        except Exception as e:
            error_msg = f"测试失败: {str(e)}"
            return error_msg, error_msg, error_msg
    
    def prepare_training_data(self, num_samples: int = 20) -> str:
        """
        准备训练数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            str: 数据准备状态
        """
        try:
            # 创建数据集
            dataset = PharmaKnowledgeDataset()
            
            # 生成训练数据
            train_data, val_data = dataset.split_dataset()[:2]
            
            # 限制样本数量
            train_samples = train_data.samples[:num_samples]
            val_samples = val_data.samples[:num_samples//4] if val_data.samples else []
            
            self.current_training_data = {
                'train': train_samples,
                'val': val_samples
            }
            
            # 显示数据样本
            sample_text = "📋 训练数据样本:\n\n"
            display_count = min(10, len(train_samples))  # 最多显示10个样本
            for i, sample in enumerate(train_samples[:display_count]):
                sample_text += f"{i+1}. {sample['question']}\n"
                sample_text += f"   类别: {sample['category']} | 难度: {sample['difficulty']}\n\n"
            
            if len(train_samples) > display_count:
                sample_text += f"... 还有 {len(train_samples) - display_count} 个样本未显示\n"
            
            sample_text += f"\n📊 总计: {len(train_samples)} 个训练样本"
            
            return sample_text
            
        except Exception as e:
            return f"数据准备失败: {str(e)}"
    
    def start_distillation_training(self, epochs: int = 5, temperature: float = 3.0, learning_rate: float = 1e-4):
        """
        开始知识蒸馏训练，并实时更新进度
        """
        if not self.current_training_data:
            yield "请先准备训练数据"
            return

        if not self.teacher_model or not self.student_model:
            yield "请先初始化模型"
            return

        try:
            yield "准备训练环境中... ⚙️"
            config = DistillationConfig(epochs=epochs, temperature=temperature, learning_rate=learning_rate)
            
            yield "正在生成教师模型输出 (这可能需要一些时间)..."
            distillation_engine = KnowledgeDistillationEngine(
                teacher_model=self.teacher_model,
                student_model=self.student_model,
                config=config
            )
            train_data = distillation_engine.prepare_pharma_knowledge(self.current_training_data['train'])
            val_data = distillation_engine.prepare_pharma_knowledge(self.current_training_data['val']) if self.current_training_data['val'] else None

            yield "数据准备完成, 初始化训练器..."
            self.trainer = DistillationTrainer(
                teacher_model=self.teacher_model,
                student_model=self.student_model,
                config=config
            )

            # 使用生成器进行训练并实时更新状态
            final_summary = None
            for update in self.trainer.train(train_data, val_data):
                if isinstance(update, str):
                    yield update
                elif isinstance(update, dict):
                    final_summary = update
            
            if final_summary:
                result_text = f"""
                🎯 知识蒸馏训练完成！
                
                📈 训练摘要:
                - 训练轮数: {final_summary['total_epochs']}
                - 平均损失: {final_summary['training_metrics']['average_total_loss']:.4f}
                - 学生模型状态: {'已训练' if final_summary['student_model_trained'] else '未训练'}
                
                💾 输出目录: {final_summary['output_directory']}
                """
                yield result_text

        except Exception as e:
            error_msg = f"❌ 训练失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg
    
    def generate_evaluation_report(self):
        """生成模型评估报告"""
        if self.evaluator is None:
            return "请先完成模型初始化"

        # 检查学生模型是否已训练
        if not self.student_model.is_trained:
            return "请先准备训练数据或训练学生模型"

        try:
            # 如果没有当前训练数据，但模型已训练，则使用默认问题进行评估
            if self.current_training_data is None:
                report_title = "### 默认测试问题评估报告\n"
                default_questions = [
                    "什么是阿司匹林？",
                    "请解释一下什么是药物相互作用？",
                    "高血压患者应该注意哪些药物？",
                    "请描述一下抗生素的正确使用方法。",
                    "什么是药物的半衰期？"
                ]
                test_data = [{"question": q} for q in default_questions]
                evaluation_results = self.evaluator.evaluate_batch(test_data)
            else:
                # 使用验证集或训练集进行评估
                if 'validation' in self.current_training_data and self.current_training_data['validation']:
                    report_title = "### 验证集评估报告\n"
                    eval_data = self.current_training_data['validation']
                else:
                    report_title = "### 训练集评估报告\n"
                    eval_data = self.current_training_data['train']
                
                evaluation_results = self.evaluator.evaluate_batch(eval_data)

            # 格式化报告
            report = report_title
            report += f"**评估时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"**评估样本数:** {len(default_questions) if self.current_training_data is None else len(eval_data)}个\n\n"
            
            report += "#### 核心指标\n"
            report += f"- **与教师模型的平均相似度:** {evaluation_results.get('similarity_to_teacher', 0):.4f}\n"
            report += f"- **平均响应质量:** {evaluation_results.get('response_quality', 0):.4f}\n"
            report += f"- **学生模型置信度提升:** {evaluation_results.get('confidence_improvement', 0):.4f}\n"
            report += f"- **响应完整性:** {evaluation_results.get('response_completeness', 0):.4f}\n"
            report += f"- **关键词覆盖率:** {evaluation_results.get('keyword_coverage', 0):.4f}\n"
            report += f"- **综合评分:** {evaluation_results.get('overall_score', 0):.4f}\n\n"
            
            report += "#### 详细指标说明\n"
            report += "- **相似度**: 学生模型回答与教师模型回答的相似程度\n"
            report += "- **响应质量**: 基于长度、置信度、内容完整性和语言规范性的综合评分\n"
            report += "- **置信度提升**: 学生模型相比初始状态的置信度改善\n"
            report += "- **响应完整性**: 学生回答对教师回答关键信息的覆盖程度\n"
            report += "- **关键词覆盖率**: 医药关键词在教师和学生回答中的一致性\n"
            
            return report
            
        except Exception as e:
            logger.error(f"生成评估报告失败: {e}", exc_info=True)
            return f"生成评估报告失败: {str(e)}"

    def evaluate_single_question_distillation(self, question: str):
        """
        对单个问题评估知识蒸馏前后的效果
        """
        if not self.teacher_model or not self.student_model:
            return "请先在“模型管理”选项卡中初始化模型", "", "", "Error"

        if not self.student_model.is_trained:
            return "当前学生模型未经训练。", "请先在“知识蒸馏”选项卡中训练学生模型，才能对比蒸馏效果。", "请先训练学生模型。", "Info"

        try:
            # 1. 获取教师模型的回答 (指导)
            teacher_response = self.teacher_model.generate_response(question).text

            # 2. 获取经过训练的学生模型的回答 (蒸馏后)
            trained_student_response = self.student_model.generate_response(question).text

            # 3. 重新创建一个未经训练的学生模型实例以获取其回答 (蒸馏前)
            untrained_student = StudentModel(model_name=self.student_model.model_name)
            untrained_student_response = untrained_student.generate_response(question).text
            
            return teacher_response, untrained_student_response, trained_student_response, "Success"

        except Exception as e:
            error_msg = f"❌ 对比分析失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, error_msg, error_msg, "Error"

    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        with gr.Blocks(title="DeepSeek 药学知识蒸馏系统", theme="soft") as app:
            gr.Markdown("""
            # 🏥 DeepSeek 蒸馏药学知识系统演示
            
            使用DeepSeek R1作为教师模型，Qwen 0.5B作为学生模型，
            通过知识蒸馏技术实现药学知识的智能迁移。
            """)
            
            with gr.Tab("🔧 模型管理"):
                gr.Markdown("### 模型初始化与测试")
                
                with gr.Row():
                    with gr.Column():
                        teacher_model_input = gr.Textbox(
                            label="教师模型名称",
                            value="deepseek-r1:latest",
                            placeholder="输入Ollama中的教师模型名称"
                        )
                        student_model_input = gr.Textbox(
                            label="学生模型名称", 
                            value="qwen2:0.5b",
                            placeholder="输入Ollama中的学生模型名称"
                        )
                        init_btn = gr.Button("🚀 初始化模型", variant="primary")
                    
                    with gr.Column():
                        init_status = gr.Textbox(
                            label="初始化状态",
                            lines=8,
                            interactive=False
                        )
                
                # 模型测试
                gr.Markdown("### 模型交互测试")
                with gr.Row():
                    test_question = gr.Textbox(
                        label="测试问题",
                        placeholder="请输入一个药学相关问题，例如：阿司匹林的副作用有哪些？",
                        lines=2
                    )
                    test_btn = gr.Button("🧪 测试模型", variant="secondary")
                
                with gr.Row():
                    teacher_response = gr.Textbox(
                        label="🎓 教师模型回答",
                        lines=25,
                        interactive=False,
                        max_lines=100
                    )
                    student_response = gr.Textbox(
                        label="👨‍🎓 学生模型回答",
                        lines=25,
                        interactive=False,
                        max_lines=100
                    )
                    evaluation_result = gr.Textbox(
                        label="📊 评估结果",
                        lines=25,
                        interactive=False,
                        max_lines=50
                    )
            
            with gr.Tab("📚 知识蒸馏"):
                gr.Markdown("### 知识蒸馏训练")
                
                with gr.Row():
                    with gr.Column():
                        num_samples = gr.Slider(
                            label="训练样本数量",
                            minimum=5,
                            maximum=100,
                            value=20,
                            step=5
                        )
                        prepare_data_btn = gr.Button("📋 准备训练数据")
                    
                    with gr.Column():
                        training_data_preview = gr.Textbox(
                            label="训练数据预览",
                            lines=10,
                            interactive=False
                        )
                
                # 训练参数设置
                gr.Markdown("### 训练参数配置")
                with gr.Row():
                    epochs_input = gr.Number(
                        label="训练轮数",
                        value=5,
                        minimum=1,
                        maximum=50
                    )
                    temperature_input = gr.Slider(
                        label="蒸馏温度",
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.1
                    )
                    learning_rate_input = gr.Number(
                        label="学习率",
                        value=0.0001
                    )
                
                train_btn = gr.Button("🎯 开始知识蒸馏训练", variant="primary")
                training_status = gr.Textbox(
                    label="训练状态",
                    lines=8,
                    interactive=False
                )
            
            with gr.Tab("📊 模型评估"):
                gr.Markdown("### 一、单问题蒸馏效果对比")
                gr.Markdown("输入一个药学问题，直观对比知识蒸馏前后学生模型回答的变化，并与教师模型的标准回答进行比较。")
                
                with gr.Row():
                    eval_question_input = gr.Textbox(label="输入评估问题", placeholder="例如：阿司匹林的副作用有哪些？", lines=2, scale=4)
                    eval_compare_btn = gr.Button("🔬 对比蒸馏效果", variant="secondary", scale=1)

                with gr.Row():
                    teacher_eval_output = gr.Textbox(label="🎓 教师模型回答 (指导)", lines=15, interactive=False)
                    untrained_student_output = gr.Textbox(label="👨‍🎓 学生模型回答 (蒸馏前)", lines=15, interactive=False)
                    trained_student_output = gr.Textbox(label="👨‍🎓 学生模型回答 (蒸馏后)", lines=15, interactive=False)
                
                eval_status_output = gr.Textbox(visible=False) # 用于状态消息传递

                gr.Markdown("---")
                gr.Markdown("### 二、批量评估报告")
                gr.Markdown("基于验证集或默认测试问题，生成包含核心性能指标（如相似度、响应质量等）的综合评估报告。")
                
                eval_btn = gr.Button("📈 生成评估报告", variant="primary")
                evaluation_report = gr.Textbox(
                    label="评估报告",
                    lines=20,
                    interactive=False
                )
            
            with gr.Tab("📈 可视化分析"):
                gr.Markdown("### 训练过程可视化")
                
                # 创建可视化组件
                visualizer = TrainingVisualizer()
                
                with gr.Row():
                    plot_type = gr.Dropdown(
                        label="图表类型",
                        choices=["训练曲线", "模型对比", "响应分析"],
                        value="训练曲线"
                    )
                    generate_plot_btn = gr.Button("📊 生成图表")
                
                plot_output = gr.Plot(label="分析图表")
            
            with gr.Tab("🔍 响应对比"):
                gr.Markdown("### 教师-学生响应对比分析")
                
                comparator = ResponseComparator()
                
                with gr.Row():
                    comparison_question = gr.Textbox(
                        label="输入问题",
                        placeholder="请输入要对比分析的问题",
                        lines=2
                    )
                    compare_btn = gr.Button("🔍 对比分析")
                
                comparison_output = gr.HTML(label="对比分析结果")
            
            # 事件绑定
            init_btn.click(
                fn=self.initialize_models,
                inputs=[teacher_model_input, student_model_input],
                outputs=init_status
            )
            
            test_btn.click(
                fn=self.test_model_interaction,
                inputs=test_question,
                outputs=[teacher_response, student_response, evaluation_result]
            )
            
            prepare_data_btn.click(
                fn=self.prepare_training_data,
                inputs=num_samples,
                outputs=training_data_preview
            )
            
            train_btn.click(
                fn=self.start_distillation_training,
                inputs=[epochs_input, temperature_input, learning_rate_input],
                outputs=training_status
            )
            
            eval_compare_btn.click(
                fn=self.evaluate_single_question_distillation,
                inputs=[eval_question_input],
                outputs=[teacher_eval_output, untrained_student_output, trained_student_output, eval_status_output]
            )

            eval_btn.click(
                fn=self.generate_evaluation_report,
                outputs=evaluation_report
            )
            
            generate_plot_btn.click(
                fn=visualizer.generate_plot,
                inputs=plot_type,
                outputs=plot_output
            )
            
            compare_btn.click(
                fn=comparator.compare_responses,
                inputs=comparison_question,
                outputs=comparison_output
            )
            
            gr.Markdown('<div style="text-align: center;">👋✨😊Powered by 信息药师 yang sitao  👋✨😊</div>')
        
        return app
    
    def launch(self, **kwargs):
        """启动应用"""
        interface = self.create_interface()
        # 启用队列以支持多并发访问
        interface.queue(
            max_size=50,                    # 最大队列长度
            default_concurrency_limit=5,   # 默认并发限制
            status_update_rate="auto"       # 状态更新频率
        )
        interface.launch(**kwargs)

# 创建应用实例
app = PharmaDistillationApp()

# 导出启动函数
def launch_app(**kwargs):
    """启动Web应用"""
    app.launch(**kwargs)

if __name__ == "__main__":
    # 配置日志（简化配置，避免与Gradio冲突）
    logging.basicConfig(level=logging.INFO)
    
    # 启动应用
    launch_app(
        server_name="0.0.0.0",
        server_port=7864,  # 更改为不同的端口
        share=False,
        debug=True
    )