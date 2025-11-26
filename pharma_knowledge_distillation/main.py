#!/usr/bin/env python3
"""
DeepSeek 药学知识蒸馏系统 - 主入口

这是一个基于知识蒸馏技术的药学知识迁移项目，
使用DeepSeek R1作为教师模型，Qwen 0.5B作为学生模型。
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.models.distillation import KnowledgeDistillationEngine, DistillationConfig
from src.data.dataset import PharmaKnowledgeDataset
from src.data.preprocessor import PharmaDataPreprocessor
from src.training.trainer import DistillationTrainer
from src.training.evaluator import ModelEvaluator
from src.web.app import PharmaDistillationApp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pharma_distillation.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class PharmaDistillationSystem:
    """药学知识蒸馏系统主类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.teacher_model = None
        self.student_model = None
        self.distillation_engine = None
        self.trainer = None
        self.evaluator = None
        
        logger.info("药学知识蒸馏系统初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"配置文件加载成功: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'models': {
                'teacher': {'name': 'deepseek-r1', 'temperature': 0.7},
                'student': {'name': 'qwen:0.5b', 'temperature': 0.8}
            },
            'distillation': {
                'temperature': 3.0,
                'alpha': 0.7,
                'beta': 0.3,
                'epochs': 10,
                'batch_size': 4
            }
        }
    
    def initialize_models(self, teacher_model: Optional[str] = None, 
                         student_model: Optional[str] = None) -> bool:
        """
        初始化模型
        
        Args:
            teacher_model: 教师模型名称
            student_model: 学生模型名称
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 获取模型配置
            teacher_name = teacher_model or self.config['models']['teacher']['name']
            student_name = student_model or self.config['models']['student']['name']
            
            # 初始化教师模型
            logger.info(f"正在初始化教师模型: {teacher_name}")
            self.teacher_model = TeacherModel(
                model_name=teacher_name,
                temperature=self.config['models']['teacher']['temperature']
            )
            
            # 初始化学生模型
            logger.info(f"正在初始化学生模型: {student_name}")
            self.student_model = StudentModel(
                model_name=student_name,
                temperature=self.config['models']['student']['temperature']
            )
            
            # 初始化评估器
            self.evaluator = ModelEvaluator(self.teacher_model, self.student_model)
            
            logger.info("✅ 模型初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            return False
    
    def quick_test(self, question: str = "阿司匹林的常见副作用有哪些？"):
        """
        快速测试模型功能
        
        Args:
            question: 测试问题
        """
        if not self.teacher_model or not self.student_model:
            logger.error("模型未初始化，请先调用 initialize_models()")
            return
        
        try:
            logger.info(f"🔍 快速测试 - 问题: {question}")
            
            # 获取教师模型回答
            teacher_response = self.teacher_model.generate_response(question)
            logger.info(f"🎓 教师模型回答: {teacher_response.text[:100]}...")
            
            # 获取学生模型回答
            student_response = self.student_model.generate_response(question)
            logger.info(f"👨‍🎓 学生模型回答: {student_response.text[:100]}...")
            
            # 评估
            evaluation = self.evaluator.evaluate_single_response(question)
            similarity = evaluation['metrics']['similarity_to_teacher']
            quality = evaluation['metrics']['response_quality']
            
            logger.info(f"📊 相似度: {similarity:.3f}, 质量: {quality:.3f}")
            
            return {
                'question': question,
                'teacher_response': teacher_response.text,
                'student_response': student_response.text,
                'similarity': similarity,
                'quality': quality
            }
            
        except Exception as e:
            logger.error(f"快速测试失败: {e}")
            return None
    
    def run_distillation(self, data_path: Optional[str] = None, 
                        num_samples: int = 20) -> Dict[str, Any]:
        """
        运行知识蒸馏
        
        Args:
            data_path: 数据文件路径
            num_samples: 样本数量
            
        Returns:
            Dict[str, Any]: 蒸馏结果
        """
        if not self.teacher_model or not self.student_model:
            logger.error("模型未初始化，请先调用 initialize_models()")
            return {}
        
        try:
            logger.info("🎯 开始知识蒸馏训练...")
            
            # 创建数据集
            if data_path and os.path.exists(data_path):
                dataset = PharmaKnowledgeDataset(data_path=data_path)
            else:
                dataset = PharmaKnowledgeDataset()
            
            # 分割数据
            train_dataset, val_dataset, _ = dataset.split_dataset()
            
            # 限制样本数量
            train_samples = train_dataset.samples[:num_samples]
            val_samples = val_dataset.samples[:min(num_samples//4, len(val_dataset.samples))]
            
            logger.info(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")
            
            # 创建蒸馏配置
            distillation_config = DistillationConfig(
                temperature=self.config['distillation']['temperature'],
                alpha=self.config['distillation']['alpha'],
                beta=self.config['distillation']['beta'],
                learning_rate=self.config['distillation']['learning_rate'],
                epochs=self.config['distillation']['epochs'],
                batch_size=self.config['distillation']['batch_size']
            )
            
            # 创建蒸馏引擎
            self.distillation_engine = KnowledgeDistillationEngine(
                teacher_model=self.teacher_model,
                student_model=self.student_model,
                config=distillation_config
            )
            
            # 准备蒸馏数据
            train_data = self.distillation_engine.prepare_pharma_knowledge(train_samples)
            val_data = self.distillation_engine.prepare_pharma_knowledge(val_samples) if val_samples else None
            
            # 执行蒸馏
            results = self.distillation_engine.distill_knowledge(train_data, val_data)
            
            logger.info("✅ 知识蒸馏训练完成")
            return results
            
        except Exception as e:
            logger.error(f"❌ 知识蒸馏失败: {e}")
            return {}
    
    def evaluate_models(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        if not self.evaluator:
            logger.error("评估器未初始化")
            return {}
        
        try:
            logger.info("📊 开始模型评估...")
            
            # 准备测试数据
            if test_data_path and os.path.exists(test_data_path):
                test_dataset = PharmaKnowledgeDataset(data_path=test_data_path)
            else:
                # 使用内置测试数据
                test_dataset = PharmaKnowledgeDataset()
            
            test_samples = test_dataset.samples[:10]  # 使用10个样本进行测试
            
            # 生成评估报告
            report = self.evaluator.generate_evaluation_report(test_samples)
            
            logger.info("✅ 模型评估完成")
            return report
            
        except Exception as e:
            logger.error(f"❌ 模型评估失败: {e}")
            return {}
    
    def launch_web_interface(self, **kwargs):
        """启动Web界面"""
        try:
            app = PharmaDistillationApp()
            
            # 如果模型已初始化，传递给Web应用
            if self.teacher_model and self.student_model:
                app.teacher_model = self.teacher_model
                app.student_model = self.student_model
                app.evaluator = self.evaluator
            
            # 启动Web应用
            app.launch(**kwargs)
            
        except Exception as e:
            logger.error(f"Web界面启动失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepSeek 药学知识蒸馏系统")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["test", "train", "eval", "web"],
        default="test",
        help="运行模式"
    )
    
    parser.add_argument(
        "--teacher-model",
        type=str,
        help="教师模型名称"
    )
    
    parser.add_argument(
        "--student-model",
        type=str,
        help="学生模型名称"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="数据文件路径"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="样本数量"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        default="阿司匹林的常见副作用有哪些？",
        help="测试问题"
    )
    
    parser.add_argument(
        "--web-port",
        type=int,
        default=7860,
        help="Web服务端口"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建系统实例
    system = PharmaDistillationSystem(config_path=args.config)
    
    # 初始化模型
    success = system.initialize_models(
        teacher_model=args.teacher_model,
        student_model=args.student_model
    )
    
    if not success:
        logger.error("系统初始化失败，退出程序")
        return
    
    # 根据模式执行相应操作
    if args.mode == "test":
        # 快速测试
        result = system.quick_test(args.question)
        if result:
            print(f"\n🎯 测试结果:")
            print(f"问题: {result['question']}")
            print(f"教师回答: {result['teacher_response']}")
            print(f"学生回答: {result['student_response']}")
            print(f"相似度: {result['similarity']:.3f}")
            print(f"质量: {result['quality']:.3f}")
    
    elif args.mode == "train":
        # 知识蒸馏训练
        results = system.run_distillation(
            data_path=args.data_path,
            num_samples=args.num_samples
        )
        
        if results:
            print(f"\n🎯 训练完成!")
            print(f"训练轮数: {results.get('total_epochs', 0)}")
            print(f"平均损失: {results.get('average_total_loss', 0):.4f}")
            print(f"输出目录: {results.get('output_dir', 'unknown')}")
    
    elif args.mode == "eval":
        # 模型评估
        report = system.evaluate_models(args.data_path)
        
        if report:
            print(f"\n📊 评估报告:")
            print(f"综合评分: {report['summary_metrics']['overall_score']:.3f}")
            print(f"相似度: {report['summary_metrics']['similarity_to_teacher']:.3f}")
            print(f"响应质量: {report['summary_metrics']['response_quality']:.3f}")
    
    elif args.mode == "web":
        # 启动Web界面
        print(f"🌐 启动Web界面，访问地址: http://localhost:{args.web_port}")
        system.launch_web_interface(
            server_name="0.0.0.0",
            server_port=args.web_port,
            share=False,
            debug=args.debug
        )
    
    logger.info("程序执行完成")

if __name__ == "__main__":
    main()