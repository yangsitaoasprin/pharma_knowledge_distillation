"""
配置管理工具
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                logger.warning(f"配置文件不存在: {self.config_path}")
                return self.get_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return self.get_default_config()
    
    def save_config(self, config: Dict[str, Any], output_path: Optional[str] = None) -> bool:
        """
        保存配置文件
        
        Args:
            config: 配置数据
            output_path: 输出路径（可选）
            
        Returns:
            bool: 是否成功保存
        """
        try:
            output_file = Path(output_path) if output_path else self.config_path
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置文件已保存: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'models': {
                'teacher': {
                    'name': 'deepseek-r1',
                    'temperature': 0.7,
                    'max_tokens': 512
                },
                'student': {
                    'name': 'qwen:0.5b',
                    'temperature': 0.8,
                    'max_tokens': 512
                }
            },
            'distillation': {
                'temperature': 3.0,
                'alpha': 0.7,
                'beta': 0.3,
                'gamma': 0.1,
                'learning_rate': 1e-4,
                'epochs': 10,
                'batch_size': 4,
                'save_interval': 5,
                'eval_interval': 2
            },
            'data': {
                'max_length': 512,
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1
            },
            'training': {
                'optimizer': 'adam',
                'weight_decay': 0.01,
                'gradient_clipping': 1.0,
                'warmup_steps': 100
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键（支持点表示法，如 'models.teacher.name'）
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            key: 配置键（支持点表示法）
            value: 配置值
            
        Returns:
            bool: 是否成功设置
        """
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"配置设置失败: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        批量更新配置
        
        Args:
            updates: 更新字典
            
        Returns:
            bool: 是否成功更新
        """
        try:
            for key, value in updates.items():
                self.set(key, value)
            
            logger.info("配置更新成功")
            return True
            
        except Exception as e:
            logger.error(f"配置更新失败: {e}")
            return False
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """
        验证配置有效性
        
        Returns:
            tuple[bool, list[str]]: (是否有效, 错误信息列表)
        """
        is_valid = True
        errors = []
        
        # 检查必需字段
        required_fields = [
            'models.teacher.name',
            'models.student.name',
            'distillation.temperature',
            'distillation.alpha',
            'distillation.beta'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                is_valid = False
                errors.append(f"缺少必需配置项: {field}")
        
        # 检查数值范围
        temperature = self.get('distillation.temperature')
        if temperature is not None and (temperature < 1.0 or temperature > 10.0):
            is_valid = False
            errors.append("蒸馏温度应在1.0-10.0之间")
        
        alpha = self.get('distillation.alpha')
        if alpha is not None and (alpha < 0.0 or alpha > 1.0):
            is_valid = False
            errors.append("alpha值应在0.0-1.0之间")
        
        beta = self.get('distillation.beta')
        if beta is not None and (beta < 0.0 or beta > 1.0):
            is_valid = False
            errors.append("beta值应在0.0-1.0之间")
        
        return is_valid, errors
    
    def export_config(self, format: str = 'json', output_path: Optional[str] = None) -> bool:
        """
        导出配置
        
        Args:
            format: 导出格式 ('json', 'yaml')
            output_path: 输出路径
            
        Returns:
            bool: 是否成功导出
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"config_export_{timestamp}.{format}"
            
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == 'yaml':
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"配置已导出: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置导出失败: {e}")
            return False

# 全局配置管理器实例
config_manager = ConfigManager()