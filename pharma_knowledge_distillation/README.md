# 🏥 DeepSeek 药学知识蒸馏系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.35+-yellow.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-✅%20运行中-brightgreen.svg)]()

## 📋 项目简介

这是一个基于知识蒸馏技术的药学知识迁移项目，使用 **DeepSeek R1** 作为教师模型，**Qwen 0.5B** 作为学生模型，通过Ollama本地部署实现药学知识的蒸馏和迁移,可用演示和训练。

### 🎯 项目目标

- **知识迁移**：将DeepSeek R1的药学专业知识迁移到更轻量的Qwen 0.5B模型
- **性能优化**：在保持较小模型体积的同时，获得接近教师模型的药学知识理解能力
- **实用应用**：实现药学问答、药物信息检索、用药指导等功能
- **可视化界面**：提供友好的Web界面进行交互和模型管理

### 🌟 项目状态

✅ **项目已优化完成** - 所有核心功能正常运行
✅ **Web界面可用** - 支持多端口同时运行 (7860-7863)
✅ **知识蒸馏就绪** - 15个药学知识样本，9个专业类别
✅ **评估系统完整** - 多维度模型性能评估

## 🏗️ 系统架构

```
pharma_knowledge_distillation/
├── src/                          # 核心源代码目录
│   ├── models/                   # 模型模块
│   │   ├── teacher_model.py     # DeepSeek R1教师模型封装
│   │   ├── student_model.py     # Qwen2 0.5B学生模型封装
│   │   └── distillation.py      # 知识蒸馏引擎
│   ├── data/                     # 数据处理模块
│   │   ├── dataset.py           # 药学数据集类
│   │   ├── preprocessor.py      # 数据预处理器
│   │   └── data_loader.py       # 数据加载器
│   ├── training/                 # 训练模块
│   │   ├── trainer.py           # 知识蒸馏训练器
│   │   ├── evaluator.py         # 模型评估器
│   │   └── loss_functions.py    # 多种损失函数
│   ├── web/                      # Web界面模块
│   │   ├── app.py               # Gradio主应用
│   │   └── components.py        # 界面组件
│   └── utils/                    # 工具模块
│       ├── config.py            # 配置管理器
│       ├── logger.py            # 日志管理器
│       └── helpers.py           # 辅助函数
├── data/                         # 数据目录
│   └── pharma_knowledge.json    # 15个药学知识样本
├── outputs/                      # 训练输出目录
├── tests/                        # 单元测试文件
├── requirements.txt             # Python依赖包列表
├── config.yaml                  # 主配置文件
├── main.py                      # 主入口文件
├── demo.py                      # 演示脚本
├── smart_trainer.py             # 智能训练器
├── trigger_training.py          # 训练触发器
├── dashboard.py                 # 训练仪表板
├── system_health_check.py       # 系统健康检查
├── check_project.py             # 项目完整性检查
└── README.md                    # 项目说明文档
```

## 🚀 快速开始

### 📋 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Ollama**: 已安装并运行
- **操作系统**: Windows/Linux/macOS

### 🛠️ 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/yangsitaoasprin/yangsitao_pharma_knowledge_distillation.git
   cd pharma_knowledge_distillation
   ```
2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```
3. **确保Ollama服务运行**

   ```bash
   # 启动Ollama服务
   ollama serve

   # 在另一个终端中拉取模型
   ollama pull deepseek-r1:latest
   ollama pull qwen2:0.5b
   ```
4. **验证模型可用性**

   ```bash
   ollama list
   ```

### 🎯 基本使用

#### 1. 快速测试

```bash
python main.py --mode test --question "阿司匹林的常见副作用有哪些？"
```

#### 2. 知识蒸馏训练

```bash
python main.py --mode train --num-samples 15
```

#### 3. 模型评估

```bash
python main.py --mode eval
```

#### 4. 启动Web界面

```bash
# 启动Web界面（支持多端口）
python main.py --mode web --web-port 7860

# 或使用演示脚本
python demo.py
```

访问地址: http://localhost:7860

### 🌐 当前运行状态

根据系统监控，项目当前运行状态：

| 服务端口 | 状态      | 功能        |
| -------- | --------- | ----------- |
| 7860     | ✅ 运行中 | 主Web界面   |
| 7861     | ✅ 运行中 | 备用Web界面 |
| 7862     | ✅ 运行中 | 备用Web界面 |
| 7863     | ✅ 运行中 | 备用Web界面 |

### 📊 系统信息

- **项目状态**: ✅ 优化完成，所有功能正常运行
- **模型配置**: DeepSeek R1 (教师) + Qwen2 0.5B (学生)
- **数据样本**: 15个药学知识问答对
- **训练就绪**: 支持知识蒸馏训练
- **评估系统**: 多维度性能评估已就绪

### 🔧 高级用法

#### 多端口Web服务

项目支持同时运行多个Web服务实例：

```bash
# 端口7860-7863已配置可用
python main.py --mode web --web-port 7861
python main.py --mode web --web-port 7862
python main.py --mode web --web-port 7863
```

#### 智能训练模式

```bash
# 使用智能训练器
python smart_trainer.py --epochs 10 --auto-tune

# 触发训练模式
python trigger_training.py --config config.yaml
```

## 📊 功能特性

### 🔧 核心功能

- **🎓 教师模型管理**: DeepSeek R1封装和交互
- **👨‍🎓 学生模型管理**: Qwen 0.5B封装和训练
- **🎯 知识蒸馏**: 多种蒸馏策略和损失函数
- **📊 性能评估**: 全面的模型评估指标
- **🌐 Web界面**: 交互式操作界面

### 📈 高级功能

- **📚 数据处理**: 药学知识数据预处理和管理
- **🔄 多任务学习**: 支持多种药学问题类型
- **📊 可视化分析**: 训练过程和结果可视化
- **🔍 模型对比**: 教师-学生模型响应对比
- **📋 实验管理**: 完整的实验跟踪和记录

## ⚙️ 配置说明

### 模型配置 (`config.yaml`)

```yaml
models:
  teacher:
    name: "deepseek-r1:latest"  # 教师模型名称
    temperature: 0.7            # 生成温度
    max_tokens: 512             # 最大token数
  student:
    name: "qwen2:0.5b"          # 学生模型名称
    temperature: 0.5            # 生成温度
    max_tokens: 256             # 最大token数

distillation:
  temperature: 3.0              # 蒸馏温度
  alpha: 0.7                    # 硬标签权重
  beta: 0.3                     # 软标签权重
  gamma: 0.1                    # 特征蒸馏权重
  learning_rate: 1e-4            # 学习率
  epochs: 10                    # 训练轮数
  batch_size: 4                 # 批次大小
```

### 训练参数

- **蒸馏温度**: 控制软标签的平滑程度，建议值 2.0-5.0
- **损失权重**: 硬标签和软标签的平衡，alpha + beta = 1.0
- **学习率**: 模型优化步长，建议值 1e-4 到 1e-5
- **批次大小**: 根据GPU内存调整，建议值 4-16

## 📈 性能指标

### 评估维度

- **🎯 相似度**: 学生模型与教师模型回答的相似程度
- **⭐ 质量评分**: 回答的专业性、准确性和完整性
- **📊 置信度**: 模型对回答的自信程度
- **🧩 完整性**: 关键信息点的覆盖程度
- **🔑 关键词覆盖**: 重要药学关键词的覆盖率

### 预期效果

- **相似度**: > 0.75
- **质量评分**: > 0.80
- **完整性**: > 0.70
- **关键词覆盖**: > 0.65
- **响应时间**: < 2秒

## 🔬 技术细节

### 知识蒸馏原理

1. **软标签蒸馏**: 使用教师模型的输出概率分布作为训练目标
2. **温度调节**: 通过温度参数控制概率分布的平滑程度
3. **损失组合**: 结合硬标签损失和软标签损失
4. **特征蒸馏**: 可选的中间层特征蒸馏

### 数据处理流程

1. **数据收集**: 药学知识问答对
2. **预处理**: 文本清洗、标准化、实体提取
3. **数据增强**: 同义改写、问题变体生成
4. **质量验证**: 数据完整性和有效性检查

## 🌐 Web界面使用

### 功能模块

1. **🔧 模型管理**: 初始化模型、测试交互
2. **📚 知识蒸馏**: 配置训练参数、启动训练
3. **📊 模型评估**: 查看评估报告和性能指标
4. **📈 可视化**: 训练曲线和模型对比图表
5. **🔍 响应对比**: 详细分析教师-学生模型差异

### 操作步骤

1. **初始化模型**: 在"模型管理"标签页初始化教师和学生模型
2. **准备数据**: 设置训练样本数量（15个药学知识样本）并准备数据
3. **配置训练**: 调整训练参数（轮数、温度、学习率）
4. **启动训练**: 开始知识蒸馏过程
5. **查看结果**: 在评估和可视化标签页查看结果

### 📊 数据样本

当前项目包含15个精心设计的药学知识样本，涵盖9个专业类别：

- **💊 药物副作用**: 阿司匹林副作用、过敏反应识别
- **🌡️ 药物储存**: 胰岛素储存方法、过期药物处理
- **📋 用药原则**: 抗生素使用、耐药性预防
- **🏥 慢性病管理**: 高血压用药注意事项
- **👶 儿科用药**: 儿童剂量计算方法
- **🧬 药理学**: 药物相互作用、半衰期概念
- **⚕️ 特殊人群**: 孕妇用药安全、妊娠分级
- **💊 药物比较**: 布洛芬vs对乙酰氨基酚
- **🧠 精神药物**: 抗抑郁药物起效时间

## 📋 示例数据

### 药学问题类型

- **💊 药物副作用**: "阿司匹林的常见副作用有哪些？"
- **🌡️ 药物储存**: "如何正确储存胰岛素？"
- **📋 用药原则**: "抗生素使用的基本原则是什么？"
- **🏥 慢性病管理**: "高血压患者用药期间需要注意什么？"
- **👶 儿科用药**: "儿童用药剂量如何计算？"

### 预期回答格式

教师模型（专业详细）:

```
阿司匹林的常见副作用包括：
1. 胃肠道反应：恶心、呕吐、胃痛
2. 出血风险：可能增加出血倾向  
3. 过敏反应：皮疹、哮喘样症状
4. 肾功能影响：长期使用可能影响肾功能

建议：饭后服用，避免空腹，如有不适及时就医。
```

学生模型（简洁实用）:

```
阿司匹林的主要副作用：
- 胃肠道不适
- 出血风险增加  
- 过敏反应

建议饭后服用，如有严重不适应及时就医。
我还在学习中，建议咨询专业医生。
```

## 🔧 故障排除

### 常见问题

1. **模型连接失败**

   - 检查Ollama服务是否运行: `ollama list`
   - 确认模型名称是否正确（使用 `deepseek-r1:latest`和 `qwen2:0.5b`）
   - 验证网络连接和端口11434
2. **训练效果不佳**

   - 检查训练样本数量（当前15个样本）
   - 调整蒸馏温度参数（推荐2.0-5.0）
   - 验证数据质量和模型响应
3. **内存不足**

   - 减小批次大小（当前配置为4）
   - 减少单次训练样本数量
   - 使用更小的模型或启用混合精度训练
4. **Web界面无法访问**

   - 检查端口是否被占用（支持7860-7863）
   - 确认防火墙设置
   - 查看日志文件 `pharma_distillation.log`
5. **多端口服务冲突**

   - 确保每个服务使用不同端口
   - 检查后台运行的服务状态
   - 使用 `python main.py --mode web --web-port 新端口`

### 日志查看

```bash
# 查看应用日志
tail -f pharma_distillation.log

# 查看训练日志
tail -f outputs/*/training_summary.json

# 查看系统健康报告
cat system_health_report_*.md
```

### 系统监控

```bash
# 运行系统健康检查
python system_health_check.py

# 查看训练仪表板
python dashboard.py

# 检查项目完整性
python check_project.py
```

## 📈 扩展开发

### 添加新功能

1. **新模型支持**: 在 `src/models/`目录添加新模型封装
2. **新损失函数**: 在 `src/training/loss_functions.py`中添加
3. **新评估指标**: 在 `src/training/evaluator.py`中扩展
4. **新界面组件**: 在 `src/web/components.py`中添加

### 自定义数据

1. 准备JSON格式的药学知识数据（参考 `data/pharma_knowledge.json`格式）
2. 使用 `PharmaDataPreprocessor`进行预处理
3. 在配置文件 `config.yaml`中指定数据路径和类别

### 训练优化

1. **超参数搜索**: 启用配置文件中的 `hyperparameter_search`选项
2. **智能训练**: 使用 `smart_trainer.py`进行自动调优
3. **批量训练**: 使用 `trigger_training.py`进行批量实验

## 🤝 贡献指南

1. **Fork项目**
2. **创建功能分支**: `git checkout -b feature/AmazingFeature`
3. **提交更改**: `git commit -m 'Add some AmazingFeature'`
4. **推送到分支**: `git push origin feature/AmazingFeature`
5. **创建Pull Request**

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **DeepSeek团队**: 提供优秀的教师模型
- **Qwen团队**: 提供轻量级的学生模型
- **Ollama团队**: 提供便捷的本地部署方案
- **开源社区**: 提供丰富的工具和框架

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- **Issues**: [GitHub Issues](https://github.com/yangsitaoasprin/yangsitao_pharma_knowledge_distillation/issues)
- **Email**: yangsitaoasprin@126.com
- **项目文档**: 查看项目根目录下的详细文档文件

### 📚 相关文档

---

**⭐ 如果这个项目对您有帮助，请给个Star支持一下！**

### 🎯 项目亮点

✅ **完整可运行**: 项目已优化完成，所有功能正常运行
✅ **多端口支持**: 同时支持4个Web服务端口(7860-7863)
✅ **专业药学知识**: 高质量药学知识样本
✅ **智能评估系统**: 多维度模型性能评估
✅ **用户友好界面**: 现代化的Gradio Web界面
✅ **模块化设计**: 易于扩展和维护的代码结构
