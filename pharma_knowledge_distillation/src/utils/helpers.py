"""
辅助工具函数
"""

import os
import json
import hashlib
import random
import string
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np

def set_random_seed(seed: int = 42):
    """
    设置随机种子以确保实验可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_id(length: int = 8) -> str:
    """
    生成随机ID
    
    Args:
        length: ID长度
        
    Returns:
        str: 随机ID
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

def generate_timestamp_id() -> str:
    """
    生成基于时间戳的ID
    
    Returns:
        str: 时间戳ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = generate_id(4)
    return f"{timestamp}_{random_suffix}"

def hash_text(text: str) -> str:
    """
    计算文本的哈希值
    
    Args:
        text: 输入文本
        
    Returns:
        str: MD5哈希值
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进级别
        
    Returns:
        bool: 是否成功保存
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        Optional[Dict[str, Any]]: 加载的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None

def create_directory(path: str) -> bool:
    """
    创建目录
    
    Args:
        path: 目录路径
        
    Returns:
        bool: 是否成功创建
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False

def get_file_size(file_path: str) -> int:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        
    Returns:
        int: 文件大小（字节）
    """
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        str: 格式化的文件大小
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def count_lines(file_path: str) -> int:
    """
    统计文件行数
    
    Args:
        file_path: 文件路径
        
    Returns:
        int: 文件行数
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def get_directory_stats(directory: str) -> Dict[str, Any]:
    """
    获取目录统计信息
    
    Args:
        directory: 目录路径
        
    Returns:
        Dict[str, Any]: 统计信息
    """
    stats = {
        'total_files': 0,
        'total_size': 0,
        'file_types': {},
        'subdirectories': []
    }
    
    try:
        for root, dirs, files in os.walk(directory):
            stats['subdirectories'].extend(dirs)
            
            for file in files:
                file_path = os.path.join(root, file)
                stats['total_files'] += 1
                stats['total_size'] += get_file_size(file_path)
                
                # 统计文件类型
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in stats['file_types']:
                    stats['file_types'][file_ext] = 0
                stats['file_types'][file_ext] += 1
    
    except Exception as e:
        print(f"获取目录统计信息失败: {e}")
    
    return stats

def clean_text(text: str) -> str:
    """
    清理文本
    
    Args:
        text: 输入文本
        
    Returns:
        str: 清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = ' '.join(text.split())
    
    # 移除特殊字符但保留基本标点
    import re
    text = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？；：""''（）【】《》]', '', text)
    
    return text.strip()

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    截断文本
    
    Args:
        text: 输入文本
        max_length: 最大长度
        
    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def calculate_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度（基于词频）
    
    Args:
        text1: 文本1
        text2: 文本2
        
    Returns:
        float: 相似度（0-1）
    """
    if not text1 or not text2:
        return 0.0
    
    # 简单的词频相似度计算
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0

def get_memory_usage() -> Dict[str, Any]:
    """
    获取内存使用情况
    
    Returns:
        Dict[str, Any]: 内存使用信息
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss,  # 常驻内存集
            'vms': memory_info.vms,  # 虚拟内存集
            'rss_mb': memory_info.rss / 1024 / 1024,  # MB
            'vms_mb': memory_info.vms / 1024 / 1024   # MB
        }
    except ImportError:
        return {'rss_mb': 0, 'vms_mb': 0}

def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"

def create_experiment_dir(base_dir: str = "experiments", 
                         experiment_name: str = None) -> str:
    """
    创建实验目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        
    Returns:
        str: 实验目录路径
    """
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # 创建子目录
    subdirs = ['checkpoints', 'logs', 'results', 'plots', 'data']
    for subdir in subdirs:
        create_directory(os.path.join(experiment_dir, subdir))
    
    return experiment_dir

def save_experiment_config(config: Dict[str, Any], 
                          experiment_dir: str, 
                          filename: str = "experiment_config.json") -> bool:
    """
    保存实验配置
    
    Args:
        config: 配置数据
        experiment_dir: 实验目录
        filename: 文件名
        
    Returns:
        bool: 是否成功保存
    """
    config_file = os.path.join(experiment_dir, filename)
    
    # 添加元数据
    config_with_meta = {
        'config': config,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'experiment_dir': experiment_dir,
            'version': '1.0'
        }
    }
    
    return save_json(config_with_meta, config_file)

def load_experiment_config(experiment_dir: str, 
                          filename: str = "experiment_config.json") -> Optional[Dict[str, Any]]:
    """
    加载实验配置
    
    Args:
        experiment_dir: 实验目录
        filename: 文件名
        
    Returns:
        Optional[Dict[str, Any]]: 配置数据
    """
    config_file = os.path.join(experiment_dir, filename)
    data = load_json(config_file)
    
    if data and 'config' in data:
        return data['config']
    
    return data

def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("🖥️  系统信息")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
    print("=" * 60)