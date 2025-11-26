# 🚀 DeepSeek 药学知识蒸馏系统 - 快速启动指南

## 🎯 30秒快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动Ollama服务（新终端）
ollama serve

# 3. 安装模型（另一个终端）
ollama pull deepseek-r1
ollama pull qwen:0.5b

# 4. 启动Web界面
python main.py --mode web

# 5. 访问 http://localhost:7860
```

---

## 📋 详细启动步骤

### 1️⃣ 环境准备

#### 系统要求
- **操作系统**: Windows 10+/Linux/macOS
- **Python**: 3.8或更高版本
- **内存**: 4GB RAM或更高
- **存储**: 2GB可用空间

#### 检查环境
```bash
# 检查Python版本
python --version

# 检查pip版本
pip --version
```

### 2️⃣ 依赖安装

#### 安装Python依赖
```bash
# 在项目根目录执行
pip install -r requirements.txt

# 如果遇到权限问题，使用：
pip install --user -r requirements.txt
```

#### 验证安装
```bash
# 检查关键依赖
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import gradio; print('Gradio:', gradio.__version__)"
```

### 3️⃣ Ollama安装和配置

#### 安装Ollama
- **Windows**: 从 https://ollama.ai 下载安装包
- **Linux/macOS**: 
  ```bash
  curl https://ollama.ai/install.sh | sh
  ```

#### 启动Ollama服务
```bash
# 在新终端窗口中启动
ollama serve

# 应该看到类似输出：
# Listening on 127.0.0.1:11434
```

#### 安装AI模型
```bash
# 在另一个终端中执行
ollama pull deepseek-r1    # 教师模型
ollama pull qwen:0.5b      # 学生模型

# 验证模型安装
ollama list
# 应该看到deepseek-r1和qwen:0.5b
```

### 4️⃣ 项目启动

#### 方法1: 使用启动脚本（推荐）
```bash
# 启动Web界面
python run_project.py --action web --port 7860

# 或快速测试
python run_project.py --action quick-test
```

#### 方法2: 直接运行主程序
```bash
# Web界面模式
python main.py --mode web --web-port 7860

# 测试模式
python main.py --mode test --question "阿司匹林的副作用有哪些？"

# 训练模式
python main.py --mode train --num-samples 20

# 评估模式
python main.py --mode eval
```

#### 方法3: 使用演示脚本
```bash
# 运行简化演示
python simple_demo.py

# 运行完整演示
python demo.py
```

### 5️⃣ 访问和使用

#### Web界面访问
- **地址**: http://localhost:7860
- **功能模块**:
  - 🔧 模型管理
  - 📚 知识蒸馏
  - 📊 模型评估
  - 📈 可视化分析
  - 🔍 响应对比

#### 命令行使用
```bash
# 测试模型交互
python main.py --mode test --question "如何储存胰岛素？"

# 开始知识蒸馏训练
python main.py --mode train --epochs 5 --temperature 3.0
```

---

## 🛠️ 常见问题解决

### ❌ Ollama连接失败
```bash
# 检查Ollama服务状态
ollama list

# 如果服务未运行，重新启动
ollama serve

# 检查防火墙设置，确保端口11434开放
```

### ❌ 模型下载失败
```bash
# 检查网络连接
ping ollama.ai

# 手动下载模型
ollama pull deepseek-r1
ollama pull qwen:0.5b

# 如果失败，尝试重启Ollama服务
```

### ❌ 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用conda
conda install pytorch torchvision torchaudio
```

### ❌ 端口被占用
```bash
# 检查端口占用
netstat -an | grep 7860

# 使用其他端口
python main.py --mode web --web-port 7861
```

---

## 🎯 使用指南

### 🔄 基本操作流程

1. **启动系统**
   ```bash
   python run_project.py --action web
   ```

2. **初始化模型**
   - 在Web界面点击"🔧 模型管理"
   - 点击"初始化模型"按钮
   - 等待模型加载完成

3. **准备数据**
   - 切换到"📚 知识蒸馏"标签
   - 设置训练样本数量
   - 点击"准备训练数据"

4. **开始训练**
   - 配置训练参数（轮数、温度等）
   - 点击"开始知识蒸馏训练"
   - 监控训练进度

5. **评估分析**
   - 切换到"📊 模型评估"标签
   - 生成评估报告
   - 查看性能指标

6. **可视化分析**
   - 切换到"📈 可视化分析"标签
   - 查看训练曲线
   - 对比模型性能

### 📋 常用命令

```bash
# 快速测试
python main.py --mode test

# 启动Web界面
python main.py --mode web

# 知识蒸馏训练
python main.py --mode train --epochs 10 --temperature 3.0

# 模型评估
python main.py --mode eval

# 项目完整性检查
python check_project.py

# 运行演示
python simple_demo.py
```

---

## 🎨 界面使用

### 🖥️ Web界面功能

#### 🔧 模型管理
- **模型初始化**: 一键初始化教师和学生模型
- **交互测试**: 测试模型问答功能
- **状态监控**: 查看模型运行状态

#### 📚 知识蒸馏
- **数据准备**: 自动准备训练数据
- **参数配置**: 可视化配置训练参数
- **训练控制**: 启动、监控训练过程
- **进度查看**: 实时查看训练进度

#### 📊 模型评估
- **性能评估**: 生成详细评估报告
- **指标分析**: 查看多维度性能指标
- **质量检查**: 评估回答质量
- **问题分析**: 识别改进点

#### 📈 可视化分析
- **训练曲线**: 显示损失和指标变化
- **模型对比**: 对比教师-学生模型性能
- **响应分析**: 分析回答特征分布
- **趋势预测**: 预测训练趋势

#### 🔍 响应对比
- **回答对比**: 并排显示教师-学生回答
- **相似度分析**: 计算文本相似度
- **质量评分**: 评估回答质量
- **改进建议**: 提供优化建议

---

## 🛡️ 最佳实践

### 📋 部署建议
1. **生产环境**: 使用Linux服务器部署
2. **开发环境**: Windows/macOS都可以
3. **内存配置**: 建议8GB以上内存
4. **存储配置**: 建议使用SSD硬盘

### 🔧 性能优化
1. **批次大小**: 根据内存调整批次大小
2. **学习率**: 使用较小的学习率(1e-4到1e-5)
3. **温度参数**: 蒸馏温度建议2.0-5.0
4. **训练轮数**: 根据数据量调整，建议5-20轮

### 📊 数据建议
1. **数据质量**: 确保数据准确性和完整性
2. **数据多样性**: 涵盖不同药学领域
3. **数据量**: 建议至少50-100个样本
4. **数据格式**: 使用标准JSON格式

---

## 🎉 成功验证

### ✅ 验证步骤
```bash
# 1. 项目完整性检查
python check_project.py

# 2. 快速测试
python main.py --mode test

# 3. 启动Web界面
python main.py --mode web

# 4. 访问 http://localhost:7860
```

### 🎊 成功标志
- [x] Web界面正常启动
- [x] 模型初始化成功
- [x] 知识蒸馏训练正常
- [x] 评估报告生成成功
- [x] 可视化图表显示正常

---

## 📞 技术支持

### 🔧 常见问题
1. **Ollama连接问题**: 检查服务状态和端口
2. **依赖问题**: 重新安装requirements.txt
3. **模型问题**: 确认模型已正确安装
4. **端口问题**: 检查端口是否被占用

### 📚 支持文档
- **README.md**: 详细项目说明
- **QUICK_START.md**: 本快速启动指南
- **COMPLETION.md**: 项目完成确认
- **DELIVERY.md**: 项目交付文档

### 💬 联系方式
- **GitHub Issues**: 问题报告和功能请求
- **项目文档**: 详细的技术文档
- **演示脚本**: 功能演示和使用示例

---

## 🎊 恭喜！成功启动！

🎉 **您已成功启动DeepSeek药学知识蒸馏系统！**

现在您可以：
1. **体验知识蒸馏**: 观察AI如何学习药学知识
2. **训练模型**: 使用自己的数据进行训练
3. **评估性能**: 分析模型的学习效果
4. **可视化分析**: 查看详细的分析图表
5. **扩展应用**: 应用到实际场景中

**下一步建议**:
- 尝试不同的训练参数
- 使用自定义的药学数据
- 探索不同的评估指标
- 扩展应用到其他医学领域

**感谢您的使用！祝项目成功！** 🚀

---

**📅 快速启动指南更新日期**: 2024年11月24日  
**✅ 验证状态**: 所有步骤经过测试，可直接使用  
**🚀 启动时间**: 5-10分钟完成完整部署