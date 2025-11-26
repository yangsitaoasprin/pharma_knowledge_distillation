"""
日志管理工具
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

class LoggerManager:
    """日志管理器"""
    
    def __init__(self, name: str = "pharma_distillation", log_dir: str = "logs"):
        """
        初始化日志管理器
        
        Args:
            name: 日志器名称
            log_dir: 日志目录
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 添加控制台处理器
        self._add_console_handler()
        
        # 添加文件处理器
        self._add_file_handler()
    
    def _add_console_handler(self):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """添加文件处理器"""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取日志器"""
        return self.logger
    
    def set_level(self, level: str or int):
        """
        设置日志级别
        
        Args:
            level: 日志级别
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 知识蒸馏训练开始")
        self.logger.info(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"⚙️  配置: {json.dumps(config, ensure_ascii=False, indent=2)}")
        self.logger.info("=" * 60)
    
    def log_training_end(self, results: Dict[str, Any]):
        """记录训练结束"""
        self.logger.info("=" * 60)
        self.logger.info("✅ 知识蒸馏训练完成")
        self.logger.info(f"📊 结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
        self.logger.info("=" * 60)
    
    def log_model_initialization(self, model_name: str, model_type: str):
        """记录模型初始化"""
        self.logger.info(f"🤖 {model_type}模型初始化: {model_name}")
    
    def log_distillation_step(self, epoch: int, loss: float, metrics: Dict[str, float]):
        """记录蒸馏步骤"""
        self.logger.info(f"📈 Epoch {epoch}: Loss={loss:.4f}, Metrics={metrics}")
    
    def log_evaluation_results(self, results: Dict[str, Any]):
        """记录评估结果"""
        self.logger.info(f"📊 评估结果: {json.dumps(results, ensure_ascii=False)}")
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        self.logger.error(f"❌ 错误 ({context}): {str(error)}", exc_info=True)
    
    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(f"⚠️  {message}")
    
    def log_info(self, message: str):
        """记录信息"""
        self.logger.info(f"ℹ️  {message}")
    
    def log_debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(f"🔍 {message}")
    
    def log_success(self, message: str):
        """记录成功信息"""
        self.logger.info(f"✅ {message}")

# 全局日志管理器实例
logger_manager = LoggerManager()

# 快捷访问函数
def get_logger(name: str = None) -> logging.Logger:
    """获取日志器"""
    if name:
        return logging.getLogger(name)
    return logger_manager.get_logger()

def log_training_start(config: Dict[str, Any]):
    """记录训练开始"""
    logger_manager.log_training_start(config)

def log_training_end(results: Dict[str, Any]):
    """记录训练结束"""
    logger_manager.log_training_end(results)

def log_model_initialization(model_name: str, model_type: str):
    """记录模型初始化"""
    logger_manager.log_model_initialization(model_name, model_type)

def log_distillation_step(epoch: int, loss: float, metrics: Dict[str, float]):
    """记录蒸馏步骤"""
    logger_manager.log_distillation_step(epoch, loss, metrics)

def log_evaluation_results(results: Dict[str, Any]):
    """记录评估结果"""
    logger_manager.log_evaluation_results(results)

def log_error(error: Exception, context: str = ""):
    """记录错误"""
    logger_manager.log_error(error, context)

def log_warning(message: str):
    """记录警告"""
    logger_manager.log_warning(message)

def log_info(message: str):
    """记录信息"""
    logger_manager.log_info(message)

def log_debug(message: str):
    """记录调试信息"""
    logger_manager.log_debug(message)

def log_success(message: str):
    """记录成功信息"""
    logger_manager.log_success(message)