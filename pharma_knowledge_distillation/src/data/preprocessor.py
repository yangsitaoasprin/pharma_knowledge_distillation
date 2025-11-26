"""
数据预处理器
处理药学知识数据的清洗、标准化和增强
"""

import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import jieba
import jieba.posseg as pseg
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class PharmaDataPreprocessor:
    """药学数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        # 加载药学专用词典
        self.pharma_terms = self._load_pharma_terms()
        
        # 初始化jieba分词器
        jieba.initialize()
        
        # 添加药学专用词汇
        for term in self.pharma_terms:
            jieba.add_word(term)
        
        logger.info(f"药学数据预处理器初始化完成，加载了 {len(self.pharma_terms)} 个专用术语")
    
    def _load_pharma_terms(self) -> List[str]:
        """加载药学专用术语"""
        terms = [
            # 药物类别
            '抗生素', '抗病毒药物', '抗真菌药物', '抗肿瘤药物', '免疫抑制剂',
            '降压药', '降糖药', '降脂药', '抗凝药', '抗血小板药',
            '镇痛药', '抗炎药', '抗过敏药', '抗抑郁药', '抗焦虑药',
            
            # 药物名称
            '阿司匹林', '布洛芬', '对乙酰氨基酚', '青霉素', '头孢',
            '阿莫西林', '奥美拉唑', '雷尼替丁', '胰岛素', '二甲双胍',
            '阿托伐他汀', '氯吡格雷', '华法林', '地高辛', '呋塞米',
            
            # 医学术语
            '药代动力学', '药效学', '生物利用度', '半衰期', '血药浓度',
            '药物相互作用', '不良反应', '副作用', '毒性', '耐药性',
            '禁忌症', '适应症', '用法用量', '给药途径', '治疗方案',
            
            # 生理系统
            '心血管系统', '呼吸系统', '消化系统', '神经系统', '内分泌系统',
            '泌尿系统', '免疫系统', '血液系统', '骨骼系统', '肌肉系统',
            
            # 疾病名称
            '高血压', '糖尿病', '冠心病', '心肌梗死', '脑卒中',
            '哮喘', '慢性阻塞性肺病', '胃溃疡', '肝炎', '肾炎',
            '抑郁症', '焦虑症', '失眠症', '癫痫', '帕金森病'
        ]
        return terms
    
    def clean_text(self, text: str) -> str:
        """
        清理文本数据
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？；：""''（）【】《》]', '', text)
        
        # 修正标点符号
        text = re.sub(r'[，,]+', '，', text)
        text = re.sub(r'[。\.]+', '。', text)
        text = re.sub(r'[？\?]+', '？', text)
        text = re.sub(r'[！\!]+', '！', text)
        
        # 移除重复的标点
        text = re.sub(r'([，。！？；：])\1+', r'\1', text)
        
        return text.strip()
    
    def normalize_medical_terms(self, text: str) -> str:
        """
        标准化医学术语
        
        Args:
            text: 原始文本
            
        Returns:
            str: 标准化后的文本
        """
        # 药物名称标准化映射
        drug_normalization = {
            '阿斯匹林': '阿司匹林',
            '阿斯匹灵': '阿司匹林',
            '乙醯胺酚': '对乙酰氨基酚',
            '扑热息痛': '对乙酰氨基酚',
            '盘尼西林': '青霉素',
            '胰岛激素': '胰岛素'
        }
        
        # 疾病名称标准化
        disease_normalization = {
            '血压高': '高血压',
            '血糖高': '糖尿病',
            '心脏病': '冠心病',
            '脑梗': '脑卒中',
            '抑郁症': '抑郁障碍'
        }
        
        # 应用标准化
        for old_term, new_term in drug_normalization.items():
            text = text.replace(old_term, new_term)
        
        for old_term, new_term in disease_normalization.items():
            text = text.replace(old_term, new_term)
        
        return text
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取医学实体
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, List[str]]: 提取的实体
        """
        entities = {
            'drugs': [],
            'diseases': [],
            'symptoms': [],
            'treatments': []
        }
        
        # 使用jieba进行分词和词性标注
        words = pseg.cut(text)
        
        for word, flag in words:
            # 基于词典匹配药物名称
            if word in self.pharma_terms:
                if any(drug in word for drug in ['药', '素', '醇', '酮', '酸']):
                    entities['drugs'].append(word)
                elif any(disease in word for disease in ['病', '症', '炎', '瘤', '中毒']):
                    entities['diseases'].append(word)
                elif any(symptom in word for symptom in ['痛', '疼', '痒', '晕', '吐', '泻']):
                    entities['symptoms'].append(word)
                elif any(treatment in word for treatment in ['治疗', '手术', '疗法', '方案']):
                    entities['treatments'].append(word)
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def classify_question_type(self, question: str) -> str:
        """
        分类问题类型
        
        Args:
            question: 问题文本
            
        Returns:
            str: 问题类型
        """
        question_lower = question.lower()
        
        # 定义问题类型关键词
        question_types = {
            'definition': ['什么是', '定义', '意思', '含义'],
            'usage': ['怎么用', '用法', '使用', '服用', '给药'],
            'side_effect': ['副作用', '不良反应', '危害', '风险'],
            'interaction': ['相互作用', '冲突', '一起用', '合用'],
            'contraindication': ['禁忌', '不能用', '不适合', '注意'],
            'dosage': ['剂量', '用量', '多少', '几片', '几粒'],
            'storage': ['储存', '保存', '存放', '保质期'],
            'comparison': ['区别', '不同', '比较', '哪个好'],
            'treatment': ['治疗', '治什么', '适应症', '用途']
        }
        
        for q_type, keywords in question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                return q_type
        
        return 'general'
    
    def augment_question(self, question: str, method: str = 'paraphrase') -> List[str]:
        """
        数据增强 - 生成问题的变体
        
        Args:
            question: 原始问题
            method: 增强方法
            
        Returns:
            List[str]: 增强后的问题列表
        """
        augmented_questions = [question]
        
        if method == 'paraphrase':
            # 同义改写
            paraphrases = {
                '什么是': ['请解释', '告诉我', '请问', '什么是'],
                '怎么用': ['如何使用', '怎样服用', '给药方法是', '用法是'],
                '副作用': ['不良反应', '危害性', '风险', '坏处'],
                '治疗': ['医治', '治愈', '治疗', '用于']
            }
            
            for original, alternatives in paraphrases.items():
                if original in question:
                    for alt in alternatives[1:]:  # 跳过第一个（原始词）
                        new_question = question.replace(original, alt)
                        augmented_questions.append(new_question)
        
        elif method == 'restructure':
            # 句式重组
            if question.startswith('什么是'):
                # 改为疑问句形式
                subject = question[3:].replace('？', '')
                restructured = f"您能解释一下{subject}吗？"
                augmented_questions.append(restructured)
            
            elif '怎么' in question:
                # 改为更正式的询问
                restructured = question.replace('怎么', '如何')
                augmented_questions.append(restructured)
        
        return list(set(augmented_questions))
    
    def validate_pharma_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证药学数据的有效性
        
        Args:
            data: 数据字典
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        is_valid = True
        errors = []
        
        # 检查必需字段
        required_fields = ['question', 'category', 'difficulty']
        for field in required_fields:
            if field not in data or not data[field]:
                is_valid = False
                errors.append(f"缺少必需字段: {field}")
        
        # 验证问题格式
        if 'question' in data:
            question = data['question']
            if not question.endswith(('？', '?')):
                is_valid = False
                errors.append("问题应该以问号结尾")
            
            if len(question) < 5 or len(question) > 200:
                is_valid = False
                errors.append("问题长度应该在5-200个字符之间")
        
        # 验证类别
        valid_categories = [
            '药物副作用', '药物储存', '用药原则', '慢性病管理',
            '儿科用药', '药理学', '药物安全', '营养补充', '精神药物'
        ]
        if 'category' in data and data['category'] not in valid_categories:
            errors.append(f"无效的类别: {data['category']}，有效类别: {valid_categories}")
        
        # 验证难度等级
        valid_difficulties = ['easy', 'medium', 'hard']
        if 'difficulty' in data and data['difficulty'] not in valid_difficulties:
            errors.append(f"无效的难度等级: {data['difficulty']}，有效等级: {valid_difficulties}")
        
        return is_valid, errors
    
    def process_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理数据
        
        Args:
            data_list: 数据列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的数据
        """
        processed_data = []
        
        for item in data_list:
            try:
                # 清理问题文本
                if 'question' in item:
                    item['question'] = self.clean_text(item['question'])
                    item['question'] = self.normalize_medical_terms(item['question'])
                
                # 提取医学实体
                if 'question' in item:
                    entities = self.extract_medical_entities(item['question'])
                    item['medical_entities'] = entities
                
                # 分类问题类型
                if 'question' in item:
                    item['question_type'] = self.classify_question_type(item['question'])
                
                # 验证数据
                is_valid, errors = self.validate_pharma_data(item)
                item['is_valid'] = is_valid
                item['validation_errors'] = errors
                
                processed_data.append(item)
                
            except Exception as e:
                logger.error(f"处理数据项时出错: {e}")
                item['is_valid'] = False
                item['processing_error'] = str(e)
                processed_data.append(item)
        
        return processed_data
    
    def generate_training_pairs(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        生成训练对（用于知识蒸馏）
        
        Args:
            data_list: 处理后的数据列表
            
        Returns:
            List[Dict[str, Any]]: 训练对数据
        """
        training_pairs = []
        
        for item in data_list:
            if not item.get('is_valid', False):
                continue
            
            base_pair = {
                'id': item.get('id', f"pair_{len(training_pairs)}"),
                'input': item['question'],
                'category': item.get('category', 'general'),
                'difficulty': item.get('difficulty', 'medium'),
                'question_type': item.get('question_type', 'general'),
                'medical_entities': item.get('medical_entities', {}),
                'metadata': {
                    'weight': item.get('weight', 1.0),
                    'source': item.get('source', 'unknown'),
                    'created_at': item.get('created_at', datetime.now().isoformat())
                }
            }
            
            # 数据增强
            if 'question' in item:
                augmented_questions = self.augment_question(item['question'], 'paraphrase')
                for i, aug_question in enumerate(augmented_questions[1:], 1):  # 跳过原始问题
                    aug_pair = base_pair.copy()
                    aug_pair['id'] = f"{base_pair['id']}_aug_{i}"
                    aug_pair['input'] = aug_question
                    aug_pair['metadata']['augmented'] = True
                    aug_pair['metadata']['original_id'] = base_pair['id']
                    training_pairs.append(aug_pair)
            
            training_pairs.append(base_pair)
        
        logger.info(f"生成了 {len(training_pairs)} 个训练对")
        return training_pairs
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str):
        """保存处理后的数据"""
        output_data = {
            'processed_samples': data,
            'metadata': {
                'total_samples': len(data),
                'processing_timestamp': datetime.now().isoformat(),
                'valid_samples': sum(1 for item in data if item.get('is_valid', False)),
                'categories': list(set(item.get('category', 'unknown') for item in data if item.get('is_valid', False))),
                'question_types': list(set(item.get('question_type', 'unknown') for item in data if item.get('is_valid', False)))
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理后的数据已保存到: {output_path}")
    
    def get_preprocessing_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取预处理统计信息"""
        valid_samples = [item for item in data if item.get('is_valid', False)]
        
        return {
            'total_samples': len(data),
            'valid_samples': len(valid_samples),
            'valid_rate': len(valid_samples) / len(data) if data else 0,
            'categories': self._count_field_values(data, 'category'),
            'question_types': self._count_field_values(data, 'question_type'),
            'difficulties': self._count_field_values(data, 'difficulty'),
            'avg_question_length': sum(len(item.get('question', '')) for item in valid_samples) / len(valid_samples) if valid_samples else 0
        }
    
    def _count_field_values(self, data: List[Dict[str, Any]], field: str) -> Dict[str, int]:
        """统计字段值分布"""
        counts = {}
        for item in data:
            value = item.get(field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts