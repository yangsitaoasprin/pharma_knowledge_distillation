"""
项目安装配置文件
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取requirements文件
def read_requirements(filename):
    """读取requirements文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

requirements = read_requirements('requirements.txt')

setup(
    name="pharma-knowledge-distillation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="DeepSeek药学知识蒸馏系统 - 使用知识蒸馏技术实现药学知识迁移",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangsitaoasprin/pharma-knowledge-distillation",
    project_urls={
        "Bug Tracker": "https://github.com/yangsitaoasprin/pharma-knowledge-distillation/issues",
        "Documentation": "https://github.com/yangsitaoasprin/pharma-knowledge-distillation/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: PyTorch",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.23.0",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pharma-distill=main:main",
            "pharma-web=src.web.app:launch_app",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.yaml",
            "*.yml", 
            "*.json",
            "*.md",
            "*.txt",
            "config.yaml",
            "requirements.txt",
            "README.md",
        ],
    },
    exclude_package_data={
        "": [
            "*.pyc",
            "__pycache__",
            ".DS_Store",
            "*.log",
            "logs/*",
            "outputs/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "knowledge-distillation",
        "pharmaceutical",
        "deepseek",
        "qwen",
        "ollama",
        "machine-learning",
        "deep-learning",
        "nlp",
        "pytorch",
        "gradio",
    ],
)