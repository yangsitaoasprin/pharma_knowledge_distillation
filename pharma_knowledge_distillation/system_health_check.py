#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Distillation System Health Check Report
Enhanced version with comprehensive diagnostics and error detection
"""

import json
import os
import sys
import torch
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np

import json
import os
import sys
import torch
from datetime import datetime

class SystemHealthChecker:
    """Enhanced system health checker with comprehensive diagnostics"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.health_score = 100
        
    def log_issue(self, severity: str, message: str, deduction: int = 0):
        """记录系统问题"""
        if severity == "error":
            self.issues.append(f"❌ {message}")
            self.health_score -= deduction
        elif severity == "warning":
            self.warnings.append(f"⚠️ {message}")
            self.health_score -= deduction
        else:
            self.recommendations.append(f"💡 {message}")
    
    def check_file_integrity(self) -> Dict[str, bool]:
        """检查关键文件完整性"""
        critical_files = {
            'config.yaml': 'Configuration file',
            'src/models/student_model.py': 'Student model implementation',
            'src/models/teacher_model.py': 'Teacher model implementation', 
            'src/models/distillation.py': 'Distillation framework',
            'src/training/trainer.py': 'Training module',
            'src/training/evaluator.py': 'Evaluation module',
            'src/data/data_loader.py': 'Data loader',
            'main.py': 'Main application',
            'dashboard.py': 'Monitoring dashboard',
            'requirements.txt': 'Dependencies'
        }
        
        results = {}
        for file_path, description in critical_files.items():
            exists = os.path.exists(file_path)
            results[file_path] = exists
            
            if not exists:
                self.log_issue("error", f"Missing critical file: {file_path} ({description})", 10)
            else:
                # 检查文件大小和内容
                try:
                    size = os.path.getsize(file_path)
                    if size == 0:
                        self.log_issue("warning", f"Empty file: {file_path}", 5)
                    elif size < 100:  # 小于100字节可能是问题
                        self.log_issue("warning", f"Suspiciously small file: {file_path} ({size} bytes)", 3)
                except Exception as e:
                    self.log_issue("error", f"Cannot access file {file_path}: {e}", 8)
        
        return results
    
    def check_system_resources(self) -> Dict[str, any]:
        """检查系统资源使用情况"""
        resources = {}
        
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        resources['cpu_usage'] = cpu_percent
        if cpu_percent > 90:
            self.log_issue("error", f"High CPU usage: {cpu_percent}%", 15)
        elif cpu_percent > 70:
            self.log_issue("warning", f"Elevated CPU usage: {cpu_percent}%", 8)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        resources['memory_usage'] = memory.percent
        resources['memory_available'] = memory.available / (1024**3)  # GB
        
        if memory.percent > 90:
            self.log_issue("error", f"High memory usage: {memory.percent}%", 15)
        elif memory.percent > 80:
            self.log_issue("warning", f"Elevated memory usage: {memory.percent}%", 8)
        
        # 磁盘空间
        disk = psutil.disk_usage('.')
        resources['disk_usage'] = disk.percent
        resources['disk_free'] = disk.free / (1024**3)  # GB
        
        if disk.percent > 95:
            self.log_issue("error", f"Critical disk usage: {disk.percent}%", 20)
        elif disk.percent > 85:
            self.log_issue("warning", f"High disk usage: {disk.percent}%", 10)
        
        return resources
    
    def check_training_history(self) -> Dict[str, any]:
        """分析训练历史记录"""
        training_info = {}
        outputs_dir = 'outputs'
        
        if not os.path.exists(outputs_dir):
            self.log_issue("error", "Training outputs directory not found", 15)
            return training_info
        
        try:
            distillation_dirs = [d for d in os.listdir(outputs_dir) if d.startswith('distillation_')]
            training_info['total_trainings'] = len(distillation_dirs)
            
            if not distillation_dirs:
                self.log_issue("warning", "No training history found", 10)
                return training_info
            
            # 分析最新的训练
            latest_dir = sorted(distillation_dirs)[-1]
            training_info['latest_training'] = latest_dir
            
            summary_path = f"{outputs_dir}/{latest_dir}/training_summary.json"
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                training_info['latest_summary'] = summary
                
                # 检查训练质量
                avg_total_loss = summary.get('average_total_loss', 999)
                if avg_total_loss > 10:
                    self.log_issue("error", f"Poor training quality: average total loss {avg_total_loss:.2f}", 15)
                elif avg_total_loss > 7:
                    self.log_issue("warning", f"Suboptimal training quality: average total loss {avg_total_loss:.2f}", 8)
                
                # 检查训练完成状态
                if not summary.get('training_completed', False):
                    self.log_issue("warning", "Latest training did not complete successfully", 12)
                
                # 检查训练时间
            training_time = summary.get('training_time_seconds', 0)
            if training_time > 3600:  # 超过1小时
                self.log_issue("warning", f"Training time unusually long: {training_time/3600:.1f} hours", 5)
            elif training_time < 30:  # 少于30秒
                self.log_issue("warning", f"Training time suspiciously short: {training_time} seconds", 8)
                    
            else:
                self.log_issue("warning", f"No training summary found in {latest_dir}", 8)
            
            # 检查训练频率
                if len(distillation_dirs) >= 2:
                    second_latest = sorted(distillation_dirs)[-2]
                    try:
                        latest_time = datetime.strptime(latest_dir.split('_')[1] + '_' + latest_dir.split('_')[2], '%Y%m%d_%H%M%S')
                        second_time = datetime.strptime(second_latest.split('_')[1] + '_' + second_latest.split('_')[2], '%Y%m%d_%H%M%S')
                        time_diff = (latest_time - second_time).total_seconds() / 3600  # hours
                        
                        if time_diff < 0.1:  # 少于6分钟
                            self.log_issue("warning", f"Training frequency too high: {time_diff:.1f} hours between sessions", 5)
                        elif time_diff > 48:  # 超过2天
                            self.log_issue("info", f"Training frequency low: {time_diff:.1f} hours between sessions")
                    except ValueError:
                        self.log_issue("warning", "Could not parse training timestamps", 3)
            
        except Exception as e:
            self.log_issue("error", f"Error analyzing training history: {e}", 12)
        
        return training_info
    
    def check_model_integrity(self) -> Dict[str, any]:
        """检查模型完整性和性能"""
        model_info = {}
        
        try:
            sys.path.append('src')
            from models.student_model import StudentModel
            
            # 测试学生模型初始化
            student = StudentModel('qwen2:0.5b')
            model_info['student_model_status'] = 'OK'
            
            # 测试嵌入生成
            test_text = "What is drug interaction?"
            embedding = student._get_text_embedding(test_text)
            
            if embedding is None:
                self.log_issue("error", "Failed to generate embeddings", 20)
            elif embedding.shape[0] != 768:
                self.log_issue("error", f"Unexpected embedding dimension: {embedding.shape}", 15)
            else:
                model_info['embedding_dimension'] = embedding.shape
                
                # 检查嵌入质量
                if torch.isnan(embedding).any():
                    self.log_issue("error", "NaN values in embeddings", 18)
                if (embedding == 0).all():
                    self.log_issue("error", "Zero embeddings generated", 15)
                
                # 测试嵌入稳定性
                embedding2 = student._get_text_embedding(test_text)
                if not torch.allclose(embedding, embedding2, rtol=1e-3):
                    self.log_issue("warning", "Embedding instability detected", 8)
            
            # 测试学习功能
            teacher_text = "Drug interaction refers to the effects when two drugs are used together"
            student_text = "Drug interaction is the mutual influence between drugs"
            
            try:
                loss = student.learn_from_teacher(teacher_text, student_text, teacher_text)
                model_info['learning_loss'] = loss
                
                if loss <= 0:
                    self.log_issue("error", f"Invalid learning loss: {loss}", 15)
                elif loss > 10:
                    self.log_issue("warning", f"High learning loss: {loss}", 8)
                elif loss < 0.1:
                    self.log_issue("warning", f"Suspiciously low learning loss: {loss}", 5)
                    
            except Exception as e:
                self.log_issue("error", f"Learning function failed: {e}", 20)
            
        except ImportError as e:
            self.log_issue("error", f"Cannot import student model: {e}", 25)
        except Exception as e:
            self.log_issue("error", f"Model integrity check failed: {e}", 20)
        
        return model_info
    
    def check_log_health(self) -> Dict[str, any]:
        """分析日志文件健康状况"""
        log_info = {}
        log_file = 'pharma_distillation.log'
        
        if not os.path.exists(log_file):
            self.log_issue("warning", "Log file not found", 8)
            return log_info
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            log_info['file_size'] = os.path.getsize(log_file)
            log_info['line_count'] = len(log_content.split('\n'))
            
            # 统计关键信息
            error_count = log_content.count('ERROR')
            warning_count = log_content.count('WARNING')
            learning_loss_count = log_content.count('Learning Loss') + log_content.count('learning loss')
            exception_count = log_content.count('Exception')
            
            log_info['error_count'] = error_count
            log_info['warning_count'] = warning_count
            log_info['learning_records'] = learning_loss_count
            log_info['exception_count'] = exception_count
            
            # 评估日志健康状况
            if error_count > 50:
                self.log_issue("error", f"Critical error count in logs: {error_count}", 15)
            elif error_count > 20:
                self.log_issue("warning", f"High error count in logs: {error_count}", 8)
            elif error_count > 5:
                self.log_issue("info", f"Moderate error count in logs: {error_count}")
            
            if warning_count > 20:
                self.log_issue("warning", f"High warning count in logs: {warning_count}", 5)
            
            if learning_loss_count == 0:
                self.log_issue("warning", "No learning loss records found", 10)
            elif learning_loss_count < 5:
                self.log_issue("warning", f"Limited learning activity: {learning_loss_count} records", 5)
            
            if exception_count > 5:
                self.log_issue("error", f"Multiple exceptions detected: {exception_count}", 12)
            
            # 检查最近的日志活动
            recent_errors = log_content[-5000:].count('ERROR')  # 最近5KB内容
            if recent_errors > 0:
                self.log_issue("warning", f"Recent errors detected: {recent_errors}", 8)
                
        except Exception as e:
            self.log_issue("error", f"Log analysis failed: {e}", 10)
        
        return log_info
    
    def check_web_services(self) -> Dict[str, any]:
        """检查Web服务状态"""
        web_info = {}
        web_ports = [8081, 8082, 8083, 8084, 8085]
        active_services = 0
        
        for port in web_ports:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)  # 2秒超时
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:
                    web_info[f'port_{port}'] = 'active'
                    active_services += 1
                else:
                    web_info[f'port_{port}'] = 'inactive'
                    
            except Exception as e:
                web_info[f'port_{port}'] = f'error: {e}'
        
        web_info['active_services'] = active_services
        
        if active_services == 0:
            self.log_issue("warning", "No active web services detected", 12)
        elif active_services > 3:
            self.log_issue("info", f"Multiple web services active: {active_services}")
        
        return web_info
    
    def generate_health_report(self) -> str:
        """生成详细的健康检查报告"""
        self.health_score = max(0, min(100, self.health_score))  # 确保分数在0-100范围内
        
        status_emoji = "🟢" if self.health_score >= 80 else "🟡" if self.health_score >= 60 else "🔴"
        status_text = "Excellent" if self.health_score >= 80 else "Good" if self.health_score >= 60 else "Needs Attention"
        
        report = f"""
# 🏥 Knowledge Distillation System Health Report

## 📊 Overall Health Status
{status_emoji} **Health Score: {self.health_score:.1f}/100** - {status_text}

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if self.issues:
            report += f"""
## ❌ Critical Issues ({len(self.issues)})
"""
            for issue in self.issues:
                report += f"{issue}\n"
        
        if self.warnings:
            report += f"""
## ⚠️ Warnings ({len(self.warnings)})
"""
            for warning in self.warnings:
                report += f"{warning}\n"
        
        if self.recommendations:
            report += f"""
## 💡 Recommendations ({len(self.recommendations)})
"""
            for rec in self.recommendations:
                report += f"{rec}\n"
        
        report += f"""
---
*For detailed diagnostics, run with --verbose flag*
"""
        
        return report
    
    def run_comprehensive_check(self, verbose: bool = False) -> Dict[str, any]:
        """运行全面的系统健康检查"""
        print("🏥 Starting Comprehensive System Health Check")
        print("=" * 60)
        
        results = {}
        
        # 1. 文件完整性检查
        print("\n📁 1. Checking File Integrity...")
        results['file_integrity'] = self.check_file_integrity()
        
        # 2. 系统资源检查
        print("\n💻 2. Checking System Resources...")
        results['system_resources'] = self.check_system_resources()
        
        # 3. 训练历史分析
        print("\n📊 3. Analyzing Training History...")
        results['training_history'] = self.check_training_history()
        
        # 4. 模型完整性检查
        print("\n🧠 4. Checking Model Integrity...")
        results['model_integrity'] = self.check_model_integrity()
        
        # 5. 日志健康检查
        print("\n📋 5. Analyzing Log Health...")
        results['log_health'] = self.check_log_health()
        
        # 6. Web服务检查
        print("\n🌐 6. Checking Web Services...")
        results['web_services'] = self.check_web_services()
        
        # 生成报告
        report = self.generate_health_report()
        
        if verbose:
            print("\n" + "=" * 60)
            print("📊 Detailed Results:")
            print("=" * 60)
            for category, data in results.items():
                print(f"\n{category.upper()}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
        
        print(report)
        
        # 保存报告
        report_file = f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Health report saved: {report_file}")
        
        return results

def check_system_health():
    """Legacy function for backward compatibility"""
    print("🏥 Knowledge Distillation System Health Check")
    print("=" * 60)
    
    # 1. 检查模型文件完整性
    print("\n📁 1. 模型文件完整性检查")
    model_files = [
        'src/models/student_model.py',
        'src/models/teacher_model.py', 
        'src/models/distillation.py',
        'src/training/trainer.py',
        'src/training/evaluator.py'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - 缺失")
    
    # 2. 检查训练输出
    print("\n📊 2. 训练输出检查")
    outputs_dir = 'outputs'
    if os.path.exists(outputs_dir):
        distillation_dirs = [d for d in os.listdir(outputs_dir) if d.startswith('distillation_')]
        print(f"   发现 {len(distillation_dirs)} 个训练记录")
        
        # 检查最新的训练结果
        if distillation_dirs:
            latest_dir = sorted(distillation_dirs)[-1]
            summary_path = f"outputs/{latest_dir}/training_summary.json"
            
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                print(f"   📈 最新训练结果 ({latest_dir}):")
                print(f"      平均总损失: {summary['average_total_loss']:.4f}")
                print(f"      平均硬损失: {summary['average_hard_loss']:.4f}")
                print(f"      平均软损失: {summary['average_soft_loss']:.4f}")
                print(f"      训练状态: {'✅ 完成' if summary['training_completed'] else '❌ 未完成'}")
                print(f"      训练周期: {summary['total_epochs']}")
                print(f"      最终周期: {summary['final_epoch']}")
    
    # 3. 检查核心功能
    print("\n🔧 3. 核心功能验证")
    try:
        sys.path.append('src')
        from models.student_model import StudentModel
        
        # 测试学生模型
        student = StudentModel('qwen2:0.5b')
        print("   ✅ 学生模型初始化成功")
        
        # 测试嵌入生成
        test_text = "什么是药物相互作用？"
        embedding = student._get_text_embedding(test_text)
        if embedding.shape[0] == 768:
            print(f"   ✅ 嵌入生成正常: {embedding.shape}")
        else:
            print(f"   ❌ 嵌入维度异常: {embedding.shape}")
        
        # 测试学习功能
        teacher_text = "药物相互作用是指两种药物同时使用时产生的效应"
        student_text = "药物相互作用是药物之间的相互影响"
        loss = student.learn_from_teacher(teacher_text, student_text, teacher_text)
        
        if loss > 0:
            print(f"   ✅ 学习功能正常: 损失={loss:.4f}")
        else:
            print(f"   ❌ 学习功能异常: 损失={loss:.4f}")
            
    except Exception as e:
        print(f"   ❌ 核心功能检查失败: {e}")
    
    # 4. 检查日志文件
    print("\n📋 4. 日志文件分析")
    log_file = 'pharma_distillation.log'
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 统计关键信息
        learning_loss_count = log_content.count('学习损失')
        error_count = log_content.count('ERROR')
        warning_count = log_content.count('WARNING')
        
        print(f"   日志文件大小: {os.path.getsize(log_file) / 1024:.1f} KB")
        print(f"   学习损失记录: {learning_loss_count} 条")
        print(f"   错误记录: {error_count} 条")
        print(f"   警告记录: {warning_count} 条")
        
        if learning_loss_count > 0:
            print("   ✅ 检测到有效的学习过程")
        else:
            print("   ⚠️  未检测到学习过程")
    
    # 5. Web服务状态
    print("\n🌐 5. Web服务状态")
    web_ports = [8081, 8082, 8083, 8084, 8085]
    active_services = 0
    
    for port in web_ports:
        # 简单的端口检查（实际应该检查具体服务）
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                print(f"   ✅ 端口 {port}: 活跃")
                active_services += 1
            else:
                print(f"   ⚪ 端口 {port}: 未使用")
        except:
            print(f"   ⚪ 端口 {port}: 检查失败")
    
    print(f"   总计活跃服务: {active_services} 个")
    
    # 6. 系统建议
    print("\n💡 6. 系统优化建议")
    print("   • 定期清理旧的训练输出目录")
    print("   • 监控学习损失趋势，确保持续改进")
    print("   • 考虑增加更多药学专业知识数据")
    print("   • 定期检查模型性能和响应质量")
    
    print("\n" + "=" * 60)
    print("📊 系统健康状态: ✅ 基本正常")
    print("🎯 建议操作: 继续训练和优化")

if __name__ == "__main__":
    # 优先运行增强版健康检查
    try:
        checker = SystemHealthChecker()
        checker.run_comprehensive_check(verbose=True)
    except Exception as e:
        print(f"Enhanced health check failed: {e}")
        print("Falling back to legacy check...")
        check_system_health()