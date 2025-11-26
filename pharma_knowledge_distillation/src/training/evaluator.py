"""
模型评估器
评估知识蒸馏效果和模型性能
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json
import re
from collections import defaultdict

from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.models.distillation import TeacherResponse, StudentResponse

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, teacher_model: TeacherModel, student_model: StudentModel):
        """
        初始化评估器
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        # 评估指标权重
        self.metric_weights = {
            'similarity': 0.4,
            'quality': 0.3,
            'confidence': 0.2,
            'completeness': 0.1
        }
        
        logger.info("模型评估器初始化完成")
    
    def evaluate_batch(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        批量评估模型性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            Dict[str, float]: 评估指标
        """
        logger.info(f"开始批量评估: {len(test_data)} 个样本")
        
        metrics = {
            'similarity_to_teacher': [],
            'response_quality': [],
            'confidence_improvement': [],
            'response_completeness': [],
            'response_length_ratio': [],
            'keyword_coverage': []
        }
        
        for idx, item in enumerate(test_data):
            question = item['question']
            teacher_response = item.get('teacher_response', '')
            
            # 获取学生模型响应
            student_response = self.student_model.generate_response(question)
            
            # 如果没有教师响应，生成教师响应用于评估
            if not teacher_response:
                teacher_response_obj = self.teacher_model.generate_response(question)
                teacher_response = teacher_response_obj.text
            
            # 计算各项指标
            similarity = self._calculate_similarity(teacher_response, student_response.text)
            quality = self._calculate_response_quality(student_response)
            completeness = self._calculate_completeness(teacher_response, student_response.text)
            length_ratio = self._calculate_length_ratio(teacher_response, student_response.text)
            keyword_coverage = self._calculate_keyword_coverage(teacher_response, student_response.text)
            
            # 置信度提升
            initial_confidence = item.get('student_confidence', 0)
            confidence_improvement = student_response.confidence - initial_confidence
            
            # 收集指标
            metrics['similarity_to_teacher'].append(similarity)
            metrics['response_quality'].append(quality)
            metrics['confidence_improvement'].append(confidence_improvement)
            metrics['response_completeness'].append(completeness)
            metrics['response_length_ratio'].append(length_ratio)
            metrics['keyword_coverage'].append(keyword_coverage)
        
        # 计算平均值
        final_metrics = {}
        for key, values in metrics.items():
            final_metrics[key] = np.mean(values) if values else 0.0
        
        # 计算综合评分
        final_metrics['overall_score'] = self._calculate_overall_score(final_metrics)
        
        logger.info(f"评估完成: 综合评分 = {final_metrics['overall_score']:.4f}")
        
        return final_metrics
    
    def evaluate_single_response(self, question: str, expected_response: str = None) -> Dict[str, Any]:
        """
        评估单个响应
        
        Args:
            question: 问题
            expected_response: 期望的响应（可选）
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 获取教师模型响应
        teacher_response = self.teacher_model.generate_response(question)
        
        # 获取学生模型响应
        student_response = self.student_model.generate_response(question)
        
        # 计算评估指标
        similarity = self._calculate_similarity(teacher_response.text, student_response.text)
        quality = self._calculate_response_quality(student_response)
        completeness = self._calculate_completeness(teacher_response.text, student_response.text)
        
        # 详细分析
        detailed_analysis = self._analyze_response_details(
            question, teacher_response, student_response
        )
        
        return {
            'question': question,
            'teacher_response': teacher_response.text,
            'student_response': student_response.text,
            'metrics': {
                'similarity_to_teacher': similarity,
                'response_quality': quality,
                'response_completeness': completeness,
                'student_confidence': student_response.confidence,
                'teacher_confidence': teacher_response.confidence
            },
            'detailed_analysis': detailed_analysis
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 基于词汇重叠的相似度
        words1 = set(self._extract_words(text1.lower()))
        words2 = set(self._extract_words(text2.lower()))
        
        # 计算Jaccard相似度
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # 计算余弦相似度（基于词频）
        all_words = union
        vec1 = [1 if word in words1 else 0 for word in all_words]
        vec2 = [1 if word in words2 else 0 for word in all_words]
        
        cosine_similarity = self._cosine_similarity(vec1, vec2)
        
        # 综合相似度
        final_similarity = (jaccard_similarity + cosine_similarity) / 2
        
        return min(1.0, max(0.0, final_similarity))
    
    def _calculate_response_quality(self, response: StudentResponse) -> float:
        """计算响应质量"""
        quality_score = 0.0
        
        # 1. 长度适宜性 (0.2)
        response_length = len(response.text)
        if 50 <= response_length <= 500:
            quality_score += 0.2
        elif 20 <= response_length < 50:
            quality_score += 0.1
        
        # 2. 置信度 (0.3)
        quality_score += response.confidence * 0.3
        
        # 3. 内容完整性检查 (0.3)
        completeness_indicators = [
            '建议', '注意', '推荐', '应该', '可以',
            '药物', '治疗', '剂量', '用法'
        ]
        
        indicator_count = sum(1 for indicator in completeness_indicators if indicator in response.text)
        completeness_score = min(1.0, indicator_count / len(completeness_indicators))
        quality_score += completeness_score * 0.3
        
        # 4. 语言规范性 (0.2)
        if '不知道' not in response.text and '不清楚' not in response.text:
            quality_score += 0.1
        
        if any(professional_term in response.text for professional_term in 
               ['药理', '机制', '临床', '疗效', '安全性']):
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _calculate_completeness(self, teacher_response: str, student_response: str) -> float:
        """计算响应完整性"""
        teacher_sentences = self._split_sentences(teacher_response)
        student_sentences = self._split_sentences(student_response)
        
        # 检查关键信息点
        key_points = self._extract_key_points(teacher_response)
        covered_points = [point for point in key_points if any(point in sent for sent in student_sentences)]
        
        completeness = len(covered_points) / len(key_points) if key_points else 0
        
        return completeness
    
    def _calculate_length_ratio(self, teacher_response: str, student_response: str) -> float:
        """计算响应长度比例"""
        teacher_len = len(teacher_response)
        student_len = len(student_response)
        
        if teacher_len == 0:
            return 0.0
        
        ratio = student_len / teacher_len
        
        # 理想比例在0.6-1.2之间
        if 0.6 <= ratio <= 1.2:
            return 1.0
        elif 0.3 <= ratio < 0.6:
            return 0.7
        elif 1.2 < ratio <= 2.0:
            return 0.8
        else:
            return 0.3
    
    def _calculate_keyword_coverage(self, teacher_response: str, student_response: str) -> float:
        """计算关键词覆盖率"""
        teacher_keywords = self._extract_medical_keywords(teacher_response)
        student_keywords = self._extract_medical_keywords(student_response)
        
        if not teacher_keywords:
            return 1.0
        
        covered_keywords = [kw for kw in teacher_keywords if kw in student_keywords]
        coverage = len(covered_keywords) / len(teacher_keywords)
        
        return coverage
    
    def _extract_medical_keywords(self, text: str) -> List[str]:
        """提取医学关键词"""
        medical_keywords = [
            # 药物相关
            '药物', '药品', '药剂', '剂量', '用法', '用量', '服用', '给药',
            # 疾病相关
            '疾病', '病症', '症状', '治疗', '治愈', '缓解',
            # 安全相关
            '副作用', '不良反应', '禁忌', '注意', '警告', '风险',
            # 效果相关
            '疗效', '效果', '作用', '功效', '功能', '益处',
            # 时间相关
            '每天', '每次', '疗程', '持续', '长期', '短期'
        ]
        
        found_keywords = [kw for kw in medical_keywords if kw in text]
        return found_keywords
    
    def _analyze_response_details(self, question: str, teacher_response: TeacherResponse, 
                                student_response: StudentResponse) -> Dict[str, Any]:
        """详细分析响应"""
        analysis = {
            'question_type': self._classify_question_type(question),
            'response_structure': self._analyze_response_structure(student_response.text),
            'medical_accuracy': self._check_medical_accuracy(student_response.text),
            'completeness_analysis': self._analyze_completeness(teacher_response.text, student_response.text),
            'improvement_suggestions': self._generate_improvement_suggestions(student_response.text)
        }
        
        return analysis
    
    def _classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['什么是', '定义', '意思']):
            return 'definition'
        elif any(word in question_lower for word in ['怎么用', '用法', '服用']):
            return 'usage'
        elif any(word in question_lower for word in ['副作用', '危害', '风险']):
            return 'side_effect'
        elif any(word in question_lower for word in ['相互', '冲突', '一起']):
            return 'interaction'
        elif any(word in question_lower for word in ['禁忌', '不能', '注意']):
            return 'contraindication'
        else:
            return 'general'
    
    def _analyze_response_structure(self, response: str) -> Dict[str, Any]:
        """分析响应结构"""
        sentences = self._split_sentences(response)
        
        structure = {
            'total_sentences': len(sentences),
            'average_sentence_length': np.mean([len(sent) for sent in sentences]) if sentences else 0,
            'has_introduction': any(word in response[:50] for word in ['首先', '一般来说', '通常']),
            'has_conclusion': any(word in response[-50:] for word in ['总之', '综上所述', '建议']),
            'has_numbering': bool(re.search(r'\d+\.|[一二三四五]、', response)),
            'paragraphs': len([p for p in response.split('\n\n') if p.strip()])
        }
        
        return structure
    
    def _check_medical_accuracy(self, response: str) -> Dict[str, Any]:
        """检查医学准确性"""
        accuracy_check = {
            'has_contradictory_terms': False,
            'has_uncertain_terms': False,
            'has_overly_confident_terms': False,
            'safety_warnings': []
        }
        
        # 检查矛盾术语
        contradictory_pairs = [
            ('安全', '危险'), ('有效', '无效'), ('推荐', '禁止')
        ]
        
        for term1, term2 in contradictory_pairs:
            if term1 in response and term2 in response:
                accuracy_check['has_contradictory_terms'] = True
                accuracy_check['safety_warnings'].append(f"发现矛盾术语: {term1} 和 {term2}")
        
        # 检查不确定性术语
        uncertain_terms = ['可能', '或许', '大概', '应该', '我觉得']
        accuracy_check['has_uncertain_terms'] = any(term in response for term in uncertain_terms)
        
        # 检查过度自信术语
        confident_terms = ['绝对', '一定', '必然', '毫无疑问']
        accuracy_check['has_overly_confident_terms'] = any(term in response for term in confident_terms)
        
        return accuracy_check
    
    def _analyze_completeness(self, teacher_response: str, student_response: str) -> Dict[str, Any]:
        """分析完整性"""
        teacher_key_points = self._extract_key_points(teacher_response)
        student_key_points = self._extract_key_points(student_response)
        
        covered_points = []
        missing_points = []
        
        for point in teacher_key_points:
            if any(self._is_similar_point(point, student_point) for student_point in student_key_points):
                covered_points.append(point)
            else:
                missing_points.append(point)
        
        return {
            'total_teacher_points': len(teacher_key_points),
            'covered_points': len(covered_points),
            'missing_points': len(missing_points),
            'coverage_rate': len(covered_points) / len(teacher_key_points) if teacher_key_points else 0,
            'covered_content': covered_points,
            'missing_content': missing_points
        }
    
    def _generate_improvement_suggestions(self, response: str) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 长度建议
        if len(response) < 30:
            suggestions.append("回答过于简短，建议提供更详细的解释")
        elif len(response) > 500:
            suggestions.append("回答过于冗长，建议精简内容")
        
        # 结构建议
        if '建议' not in response and '注意' not in response:
            suggestions.append("建议增加具体的建议或注意事项")
        
        # 专业性建议
        if not any(term in response for term in ['药物', '治疗', '剂量', '用法']):
            suggestions.append("建议增加更多专业的药学术语")
        
        return suggestions
    
    def _extract_words(self, text: str) -> List[str]:
        """提取词汇"""
        # 简单的中文分词（实际应用中应使用专业分词工具）
        import re
        
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text)
        
        # 常见的中文药学词汇（简单的词典）
        common_words = [
            '阿司匹林', '抗炎', '药物', '镇痛', '解热', '作用', '治疗', '预防',
            '心脏病', '中风', '炎症', '血小板', '聚集', '抑制', '效果', '剂量',
            '用法', '副作用', '禁忌', '适应症', '机制', '药理', '临床', '应用',
            '常见', '主要', '用于', '可以', '具有', '能够', '需要', '应该',
            '注意', '重要', '必须', '建议', '避免', '减少', '增加', '降低'
        ]
        
        # 移除标点符号
        clean_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
        
        # 简单的最大匹配分词
        chinese_words = []
        i = 0
        while i < len(clean_text):
            # 尝试找到最长的匹配词
            matched = False
            for length in range(min(6, len(clean_text) - i), 0, -1):
                word = clean_text[i:i + length]
                if word in common_words and length > 1:
                    chinese_words.append(word)
                    i += length
                    matched = True
                    break
            
            if not matched:
                # 如果没有匹配到常见词，尝试2字符词组
                if i + 2 <= len(clean_text):
                    word = clean_text[i:i + 2]
                    chinese_words.append(word)
                    i += 2
                else:
                    i += 1
        
        # 合并结果
        words = english_words + chinese_words
        return [word for word in words if len(word.strip()) > 1]
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 按标点符号分割
        sentences = re.split(r'[。！？；]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_key_points(self, text: str) -> List[str]:
        """提取关键点"""
        sentences = self._split_sentences(text)
        key_points = []
        
        for sentence in sentences:
            # 提取包含关键信息的句子
            if any(keyword in sentence for keyword in ['应该', '建议', '注意', '重要', '必须']):
                key_points.append(sentence)
        
        return key_points
    
    def _is_similar_point(self, point1: str, point2: str) -> bool:
        """检查两个点是否相似"""
        # 简单的相似度检查
        words1 = set(self._extract_words(point1.lower()))
        words2 = set(self._extract_words(point2.lower()))
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.3
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        similarity = metrics.get('similarity_to_teacher', 0)
        quality = metrics.get('response_quality', 0)
        confidence = metrics.get('confidence_improvement', 0)
        completeness = metrics.get('response_completeness', 0)
        
        overall_score = (
            similarity * self.metric_weights['similarity'] +
            quality * self.metric_weights['quality'] +
            confidence * self.metric_weights['confidence'] +
            completeness * self.metric_weights['completeness']
        )
        
        return overall_score
    
    def generate_evaluation_report(self, test_data: List[Dict[str, Any]], 
                                 output_path: str = None) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            test_data: 测试数据
            output_path: 输出路径（可选）
            
        Returns:
            Dict[str, Any]: 详细评估报告
        """
        logger.info("生成评估报告...")
        
        # 批量评估
        batch_metrics = self.evaluate_batch(test_data)
        
        # 详细分析每个样本
        detailed_results = []
        for item in test_data:
            single_eval = self.evaluate_single_response(
                item['question'], 
                item.get('teacher_response', '')
            )
            detailed_results.append(single_eval)
        
        # 生成报告
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'summary_metrics': batch_metrics,
            'detailed_results': detailed_results,
            'model_comparison': self._compare_models(),
            'improvement_recommendations': self._generate_recommendations(detailed_results),
            'statistics': self._calculate_statistics(detailed_results)
        }
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"评估报告已保存: {output_path}")
        
        return report
    
    def _compare_models(self) -> Dict[str, Any]:
        """比较教师和学生模型"""
        return {
            'teacher_model_info': self.teacher_model.get_model_info(),
            'student_model_info': self.student_model.get_model_info(),
            'model_size_ratio': 0.5,  # 假设学生模型是教师模型的一半大小
            'performance_gap': 'To be measured',  # 需要实际测试数据
            'distillation_effectiveness': 'Evaluated'
        }
    
    def _generate_recommendations(self, detailed_results: List[Dict]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 分析常见问题
        low_similarity_count = sum(1 for result in detailed_results 
                                 if result['metrics']['similarity_to_teacher'] < 0.5)
        
        low_quality_count = sum(1 for result in detailed_results 
                              if result['metrics']['response_quality'] < 0.5)
        
        if low_similarity_count > len(detailed_results) * 0.3:
            recommendations.append("学生模型与教师模型相似度较低，建议增加训练数据量或调整蒸馏温度")
        
        if low_quality_count > len(detailed_results) * 0.2:
            recommendations.append("响应质量有待提高，建议优化模型架构或增加训练轮数")
        
        recommendations.append("继续监控模型性能，定期进行性能评估")
        recommendations.append("考虑使用更多样化的训练数据以提高泛化能力")
        
        return recommendations
    
    def _calculate_statistics(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """计算统计信息"""
        similarities = [r['metrics']['similarity_to_teacher'] for r in detailed_results]
        qualities = [r['metrics']['response_quality'] for r in detailed_results]
        confidences = [r['metrics']['student_confidence'] for r in detailed_results]
        
        return {
            'similarity_stats': {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            },
            'quality_stats': {
                'mean': np.mean(qualities),
                'std': np.std(qualities),
                'min': np.min(qualities),
                'max': np.max(qualities)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'total_samples': len(detailed_results)
        }