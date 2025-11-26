#!/usr/bin/env python3
"""
项目完整性检查脚本
验证项目结构和文件是否完整
"""

import os
import sys
from pathlib import Path
import json
import yaml

def check_file_exists(file_path):
    """检查文件是否存在"""
    return Path(file_path).exists()

def check_directory_exists(dir_path):
    """检查目录是否存在"""
    return Path(dir_path).is_dir()

def check_json_file(file_path):
    """检查JSON文件格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, "Valid JSON"
    except Exception as e:
        return False, str(e)

def check_yaml_file(file_path):
    """检查YAML文件格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True, "Valid YAML"
    except Exception as e:
        return False, str(e)

def check_python_file(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        return True, "Valid Python syntax"
    except Exception as e:
        return False, str(e)

def main():
    """主检查函数"""
    print("🔍 检查项目完整性...")
    print("=" * 60)
    
    # 定义需要检查的文件和目录
    required_files = [
        "main.py",
        "config.yaml",
        "requirements.txt", 
        "README.md",
        "setup.py",
        "run_project.py",
        "data/pharma_knowledge.json",
        "src/models/teacher_model.py",
        "src/models/student_model.py",
        "src/models/distillation.py",
        "src/data/dataset.py",
        "src/data/preprocessor.py",
        "src/data/data_loader.py",
        "src/training/trainer.py",
        "src/training/evaluator.py",
        "src/training/loss_functions.py",
        "src/web/app.py",
        "src/web/components.py",
        "src/utils/config.py",
        "src/utils/logger.py",
        "src/utils/helpers.py",
        "tests/test_models.py",
        "tests/test_data.py",
        "tests/test_training.py"
    ]
    
    required_directories = [
        "src",
        "src/models",
        "src/data", 
        "src/training",
        "src/web",
        "src/utils",
        "data",
        "tests",
        "outputs"
    ]
    
    # 检查目录
    print("📁 检查目录结构...")
    all_dirs_exist = True
    for dir_path in required_directories:
        if check_directory_exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - 目录不存在")
            all_dirs_exist = False
    
    print()
    
    # 检查文件
    print("📄 检查文件...")
    all_files_exist = True
    file_checks = {}
    
    for file_path in required_files:
        if check_file_exists(file_path):
            file_checks[file_path] = {"exists": True, "error": None}
            
            # 根据文件类型进行额外检查
            if file_path.endswith('.json'):
                valid, error = check_json_file(file_path)
                if not valid:
                    file_checks[file_path]["error"] = f"JSON格式错误: {error}"
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                valid, error = check_yaml_file(file_path)
                if not valid:
                    file_checks[file_path]["error"] = f"YAML格式错误: {error}"
            elif file_path.endswith('.py'):
                valid, error = check_python_file(file_path)
                if not valid:
                    file_checks[file_path]["error"] = f"Python语法错误: {error}"
            
            status = "✅" if file_checks[file_path]["error"] is None else "⚠️"
            print(f"{status} {file_path}")
            if file_checks[file_path]["error"]:
                print(f"   {file_checks[file_path]['error']}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            file_checks[file_path] = {"exists": False, "error": "文件不存在"}
            all_files_exist = False
    
    print()
    
    # 统计结果
    total_files = len(required_files)
    existing_files = sum(1 for check in file_checks.values() if check["exists"])
    valid_files = sum(1 for check in file_checks.values() if check["exists"] and check["error"] is None)
    
    print("📊 检查结果统计:")
    print(f"总文件数: {total_files}")
    print(f"存在文件: {existing_files}")
    print(f"有效文件: {valid_files}")
    print(f"完成度: {existing_files/total_files*100:.1f}%")
    print(f"质量度: {valid_files/total_files*100:.1f}%")
    
    print()
    
    # 项目完整性评估
    if all_dirs_exist and all_files_exist:
        print("🎉 项目结构完整！")
        
        if valid_files == total_files:
            print("✨ 所有文件格式正确，项目可以正常运行！")
            return 0
        else:
            print("⚠️  项目结构完整，但部分文件存在格式问题，需要修复")
            return 1
    else:
        print("❌ 项目结构不完整，缺少必要的文件或目录")
        
        # 列出缺失的文件
        missing_files = [f for f, check in file_checks.items() if not check["exists"]]
        if missing_files:
            print("\n📋 缺失的文件:")
            for file in missing_files:
                print(f"  - {file}")
        
        return 2

def generate_project_summary():
    """生成项目摘要"""
    print("\n📋 项目摘要:")
    print("=" * 60)
    
    # 统计代码行数
    python_files = list(Path(".").rglob("*.py"))
    total_lines = 0
    
    for py_file in python_files:
        if "__pycache__" not in str(py_file):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
            except Exception:
                pass
    
    print(f"Python文件数: {len(python_files)}")
    print(f"总代码行数: {total_lines}")
    
    # 统计其他文件
    json_files = list(Path(".").rglob("*.json"))
    yaml_files = list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.yml"))
    
    print(f"JSON文件数: {len(json_files)}")
    print(f"YAML文件数: {len(yaml_files)}")
    
    # 项目结构
    print("\n🏗️  项目结构:")
    print("├── src/                    # 源代码目录")
    print("│   ├── models/             # 模型模块")
    print("│   ├── data/               # 数据处理模块")
    print("│   ├── training/           # 训练模块")
    print("│   ├── web/                # Web界面模块")
    print("│   └── utils/              # 工具模块")
    print("├── data/                   # 数据目录")
    print("├── tests/                  # 测试文件")
    print("├── outputs/                # 输出目录")
    print("├── main.py                 # 主入口文件")
    print("├── config.yaml             # 配置文件")
    print("├── requirements.txt        # 依赖包列表")
    print("└── README.md               # 项目说明")

if __name__ == "__main__":
    exit_code = main()
    generate_project_summary()
    
    print(f"\n🔚 检查完成，退出码: {exit_code}")
    sys.exit(exit_code)