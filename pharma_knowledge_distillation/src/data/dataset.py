"""
药学知识数据集类
处理药学相关的问答数据和知识库
"""

import json
import logging
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
import random
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class PharmaKnowledgeDataset(Dataset):
    """药学知识数据集"""
    
    def __init__(self, data_path: str = None, data_list: List[Dict] = None, 
                 max_length: int = 512, split: str = 'train'):
        """
        初始化药学知识数据集
        
        Args:
            data_path: 数据文件路径
            data_list: 数据列表（直接提供数据）
            max_length: 最大序列长度
            split: 数据集分割类型 ('train', 'val', 'test')
        """
        self.data_path = data_path
        self.max_length = max_length
        self.split = split
        self.samples = []
        
        if data_list:
            self.samples = data_list
        elif data_path and os.path.exists(data_path):
            self._load_data_from_file(data_path)
        else:
            self._generate_sample_data()
        
        # 数据预处理和增强
        self._preprocess_data()
        
        logger.info(f"药学知识数据集初始化完成: {len(self.samples)} 个样本，类型: {split}")
    
    def _load_data_from_file(self, file_path: str):
        """从文件加载数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.samples = data
            elif isinstance(data, dict) and 'samples' in data:
                self.samples = data['samples']
            else:
                raise ValueError("数据格式不正确，期望列表或包含'samples'键的字典")
            
            logger.info(f"从 {file_path} 加载了 {len(self.samples)} 个样本")
            
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            self._generate_sample_data()
    
    def _generate_sample_data(self):
        """生成示例药学知识数据"""
        logger.info("生成示例药学知识数据...")
        
        # 药学专业领域数据样本 - 扩展至100+样本
        sample_questions = [
            # 基础药理学 (20个)
            {
                "question": "阿司匹林的常见副作用有哪些？",
                "category": "药物副作用",
                "difficulty": "easy",
                "keywords": ["阿司匹林", "副作用", "胃肠道", "出血"]
            },
            {
                "question": "什么是药物的半衰期？",
                "category": "药理学基础",
                "difficulty": "easy",
                "keywords": ["半衰期", "药物代谢", "血药浓度", "给药间隔"]
            },
            {
                "question": "什么是药物的生物利用度？",
                "category": "药理学基础",
                "difficulty": "medium",
                "keywords": ["生物利用度", "药物吸收", "口服给药", "首过效应"]
            },
            {
                "question": "什么是首过效应？",
                "category": "药理学基础",
                "difficulty": "medium",
                "keywords": ["首过效应", "肝脏代谢", "口服给药", "生物利用度"]
            },
            {
                "question": "药物代谢的主要器官是什么？",
                "category": "药理学基础",
                "difficulty": "easy",
                "keywords": ["药物代谢", "肝脏", "CYP450", "代谢途径"]
            },
            {
                "question": "什么是药物的血浆蛋白结合率？",
                "category": "药理学基础",
                "difficulty": "hard",
                "keywords": ["血浆蛋白结合", "游离药物", "药物分布", "相互作用"]
            },
            {
                "question": "药物的排泄途径主要有哪些？",
                "category": "药理学基础",
                "difficulty": "medium",
                "keywords": ["药物排泄", "肾脏", "胆汁", "粪便", "尿液"]
            },
            {
                "question": "什么是药物的治疗指数？",
                "category": "药理学基础",
                "difficulty": "hard",
                "keywords": ["治疗指数", "LD50", "ED50", "安全范围"]
            },
            {
                "question": "什么是药物的效价强度？",
                "category": "药理学基础",
                "difficulty": "hard",
                "keywords": ["效价强度", "剂量反应", "药物活性", "EC50"]
            },
            {
                "question": "什么是药物的最大效应（Emax）？",
                "category": "药理学基础",
                "difficulty": "hard",
                "keywords": ["最大效应", "Emax", "效能", "剂量反应曲线"]
            },
            
            # 药物化学 (15个)
            {
                "question": "什么是药物的构效关系？",
                "category": "药物化学",
                "difficulty": "hard",
                "keywords": ["构效关系", "化学结构", "生物活性", "结构改造"]
            },
            {
                "question": "什么是前药？",
                "category": "药物化学",
                "difficulty": "medium",
                "keywords": ["前药", "生物转化", "药物设计", "生物利用度"]
            },
            {
                "question": "什么是手性药物？",
                "category": "药物化学",
                "difficulty": "hard",
                "keywords": ["手性药物", "对映体", "立体选择性", "药效差异"]
            },
            {
                "question": "药物的溶解度对药效有什么影响？",
                "category": "药物化学",
                "difficulty": "medium",
                "keywords": ["溶解度", "药物吸收", "生物利用度", "制剂设计"]
            },
            {
                "question": "什么是药物的脂水分配系数？",
                "category": "药物化学",
                "difficulty": "hard",
                "keywords": ["脂水分配系数", "LogP", "膜通透性", "药物吸收"]
            },
            
            # 药剂学 (15个)
            {
                "question": "如何正确储存胰岛素？",
                "category": "药剂学",
                "difficulty": "medium",
                "keywords": ["胰岛素", "储存", "温度", "冷藏", "稳定性"]
            },
            {
                "question": "什么是缓释制剂？",
                "category": "药剂学",
                "difficulty": "medium",
                "keywords": ["缓释制剂", "控释", "长效", "血药浓度"]
            },
            {
                "question": "什么是靶向制剂？",
                "category": "药剂学",
                "difficulty": "hard",
                "keywords": ["靶向制剂", "被动靶向", "主动靶向", "药物递送"]
            },
            {
                "question": "为什么有些药物需要包衣？",
                "category": "药剂学",
                "difficulty": "medium",
                "keywords": ["包衣", "肠溶", "掩味", "稳定性", "控释"]
            },
            {
                "question": "什么是纳米药物？",
                "category": "药剂学",
                "difficulty": "hard",
                "keywords": ["纳米药物", "纳米粒", "药物载体", "靶向治疗"]
            },
            
            # 药物治疗学 (20个)
            {
                "question": "抗生素使用的基本原则是什么？",
                "category": "药物治疗学",
                "difficulty": "medium",
                "keywords": ["抗生素", "合理用药", "耐药性", "疗程", "选择"]
            },
            {
                "question": "如何预防药物耐药性？",
                "category": "药物治疗学",
                "difficulty": "hard",
                "keywords": ["耐药性", "预防", "合理用药", "抗生素管理", "监测"]
            },
            {
                "question": "什么是抗菌谱？",
                "category": "药物治疗学",
                "difficulty": "medium",
                "keywords": ["抗菌谱", "抗生素", "细菌", "敏感性"]
            },
            {
                "question": "什么是抗生素的后效应？",
                "category": "药物治疗学",
                "difficulty": "hard",
                "keywords": ["抗生素后效应", "PAE", "药效学", "给药方案"]
            },
            {
                "question": "抗高血压药物有哪些类别？",
                "category": "药物治疗学",
                "difficulty": "medium",
                "keywords": ["抗高血压", "ACEI", "ARB", "钙通道阻滞剂", "利尿剂"]
            },
            
            # 药物分析 (10个)
            {
                "question": "什么是HPLC？",
                "category": "药物分析",
                "difficulty": "medium",
                "keywords": ["HPLC", "高效液相色谱", "药物分析", "含量测定"]
            },
            {
                "question": "什么是药物的含量均匀度？",
                "category": "药物分析",
                "difficulty": "medium",
                "keywords": ["含量均匀度", "制剂质量", "剂量单位", "变异性"]
            },
            {
                "question": "什么是药物的溶出度？",
                "category": "药物分析",
                "difficulty": "hard",
                "keywords": ["溶出度", "生物利用度", "体外释放", "质量控制"]
            },
            
            # 临床药学 (15个)
            {
                "question": "儿童用药剂量如何计算？",
                "category": "临床药学",
                "difficulty": "hard",
                "keywords": ["儿童用药", "剂量计算", "体重", "年龄", "体表面积"]
            },
            {
                "question": "孕妇用药需要注意什么？",
                "category": "临床药学",
                "difficulty": "hard",
                "keywords": ["孕妇用药", "胎儿安全", "禁忌药物", "妊娠分级", "FDA分级"]
            },
            {
                "question": "老年人用药有哪些特点？",
                "category": "临床药学",
                "difficulty": "hard",
                "keywords": ["老年人", "药物代谢", "多重用药", "不良反应", "剂量调整"]
            },
            {
                "question": "药物剂量如何根据肝肾功能调整？",
                "category": "临床药学",
                "difficulty": "hard",
                "keywords": ["剂量调整", "肝功能", "肾功能", "药物代谢", "清除率"]
            },
            {
                "question": "什么是治疗药物监测？",
                "category": "临床药学",
                "difficulty": "hard",
                "keywords": ["治疗药物监测", "TDM", "血药浓度", "个体化用药", "剂量优化"]
            },
            
            # 药物安全 (10个)
            {
                "question": "如何识别药物过敏反应？",
                "category": "药物安全",
                "difficulty": "medium",
                "keywords": ["药物过敏", "皮疹", "呼吸困难", "过敏反应", "过敏性休克"]
            },
            {
                "question": "什么是药物的禁忌症？",
                "category": "药物安全",
                "difficulty": "easy",
                "keywords": ["禁忌症", "药物安全", "禁用情况", "用药风险", "绝对禁忌"]
            },
            {
                "question": "如何正确处理过期药物？",
                "category": "药物安全",
                "difficulty": "easy",
                "keywords": ["过期药物", "处理方式", "环境保护", "安全处置", "回收"]
            },
            {
                "question": "什么是药物不良反应？",
                "category": "药物安全",
                "difficulty": "medium",
                "keywords": ["不良反应", "副作用", "药物安全", "监测", "报告"]
            },
            
            # 药物相互作用 (10个)
            {
                "question": "什么是药物相互作用？",
                "category": "药物相互作用",
                "difficulty": "easy",
                "keywords": ["药物相互作用", "药效", "副作用", "联合用药", "机制"]
            },
            {
                "question": "中药和西药可以同时服用吗？",
                "category": "药物相互作用",
                "difficulty": "medium",
                "keywords": ["中药", "西药", "联合用药", "药物相互作用", "时间间隔"]
            },
            {
                "question": "华法林与哪些药物有相互作用？",
                "category": "药物相互作用",
                "difficulty": "hard",
                "keywords": ["华法林", "抗凝药", "相互作用", "出血风险", "INR"]
            },
            {
                "question": "什么是酶诱导和酶抑制？",
                "category": "药物相互作用",
                "difficulty": "hard",
                "keywords": ["酶诱导", "酶抑制", "CYP450", "药物代谢", "相互作用"]
            }
        ]
        
        # 为每个问题生成多个变体
        for question_data in sample_questions:
            base_sample = {
                "id": f"pharma_{len(self.samples)}",
                "question": question_data["question"],
                "category": question_data["category"],
                "difficulty": question_data["difficulty"],
                "keywords": question_data["keywords"],
                "created_at": datetime.now().isoformat(),
                "source": "generated"
            }
            
            self.samples.append(base_sample)
            
            # 生成相似问题变体 - 药学专业领域全覆盖
            variants_map = {
                "药理学基础": [
                    "什么是药物的受体理论？",
                    "激动剂和拮抗剂有什么区别？",
                    "什么是药物的效能和效价？",
                    "药物的选择性是什么意思？",
                    "什么是药物耐受性？",
                    "药物依赖性是如何产生的？",
                    "什么是药物的副作用和毒性反应？",
                    "药物的吸收机制有哪些？",
                    "什么是药物的分布容积？",
                    "药物的清除率如何计算？",
                    "什么是药物的药代动力学？",
                    "什么是药物的药效动力学？",
                    "如何理解药物的时间-效应关系？",
                    "什么是药物的剂量-效应关系？",
                    "什么是药物的双相效应？"
                ],
                "药物化学": [
                    "什么是药物的化学稳定性？",
                    "如何改善药物的溶解性？",
                    "什么是药物的晶型？",
                    "药物的多晶型对药效有影响吗？",
                    "什么是药物的盐型？",
                    "如何选择合适的药物盐型？",
                    "什么是药物的共晶？",
                    "药物的化学修饰有哪些方法？",
                    "什么是药物的生物电子等排体？",
                    "如何提高药物的靶向性？",
                    "什么是药物的代谢稳定性？",
                    "如何设计前药？",
                    "什么是药物的脂溶性？",
                    "药物的分子量对药效有影响吗？",
                    "什么是药物的极性表面积？"
                ],
                "药剂学": [
                    "什么是药物的释放机制？",
                    "如何设计控释制剂？",
                    "什么是透皮给药系统？",
                    "纳米粒药物载体有哪些优势？",
                    "什么是脂质体药物？",
                    "如何制备微球制剂？",
                    "什么是药物的稳定性试验？",
                    "如何评估药物制剂的质量？",
                    "什么是药物的相容性？",
                    "如何优化药物的口感？",
                    "什么是药物的掩味技术？",
                    "如何防止药物的光降解？",
                    "什么是药物的氧化稳定性？",
                    "如何设计儿童友好的药物剂型？",
                    "什么是口腔崩解片？"
                ],
                "药物治疗学": [
                    "如何选择合适的抗菌药物？",
                    "什么是抗菌药物的PK/PD参数？",
                    "如何评估抗菌治疗的疗效？",
                    "什么是抗真菌药物的分类？",
                    "抗病毒药物的作用机制有哪些？",
                    "如何选择抗高血压药物？",
                    "什么是抗凝治疗的监测指标？",
                    "如何优化糖尿病药物治疗？",
                    "什么是肿瘤化疗方案的设计？",
                    "如何评估化疗药物的毒性？",
                    "什么是免疫抑制剂的应用？",
                    "如何选择合适的镇痛药物？",
                    "什么是抗炎药物的分类？",
                    "如何优化精神疾病药物治疗？",
                    "什么是药物治疗的个体化？"
                ],
                "药物分析": [
                    "什么是药物的含量测定？",
                    "如何建立药物的分析方法？",
                    "什么是药物的有关物质？",
                    "如何控制药物中的杂质？",
                    "什么是药物的溶出度试验？",
                    "如何评估药物的稳定性？",
                    "什么是药物的鉴别试验？",
                    "如何测定药物中的残留溶剂？",
                    "什么是药物的水分测定？",
                    "如何控制药物的微生物限度？",
                    "什么是药物的无菌检查？",
                    "如何测定药物的粒度分布？",
                    "什么是药物的比表面积？",
                    "如何评估药物的晶型？",
                    "什么是药物的纯度分析？"
                ],
                "临床药学": [
                    "如何优化给药方案？",
                    "什么是药物治疗的监测？",
                    "如何评估患者的用药依从性？",
                    "什么是药物重整？",
                    "如何识别药物治疗问题？",
                    "什么是药物治疗的成本效益？",
                    "如何评估药物相关的住院？",
                    "什么是药物信息的评价？",
                    "如何开展用药教育？",
                    "什么是药物治疗的循证医学？",
                    "如何评估药物治疗的结局？",
                    "什么是药物治疗的药物经济学？",
                    "如何优化多重用药？",
                    "什么是药物治疗的基因检测？",
                    "如何评估药物治疗的个体差异？"
                ],
                "药物安全": [
                    "如何识别和预防用药错误？",
                    "什么是药物警戒？",
                    "如何报告药物不良反应？",
                    "什么是药物安全信号？",
                    "如何评估药物的风险效益？",
                    "什么是药物的黑框警告？",
                    "如何识别药物的安全性问题？",
                    "什么是药物的召回制度？",
                    "如何评估特殊人群用药安全？",
                    "什么是药物的安全监测？",
                    "如何预防药物相关的伤害？",
                    "什么是药物的安全标签？",
                    "如何评估药物的安全信息？",
                    "什么是药物的上市后安全性研究？",
                    "如何建立药物安全文化？"
                ],
                "药物相互作用": [
                    "什么是药物代谢酶的诱导？",
                    "什么是药物代谢酶的抑制？",
                    "如何预测药物相互作用？",
                    "什么是药物转运体的相互作用？",
                    "如何评估药物相互作用的风险？",
                    "什么是药物相互作用的临床意义？",
                    "如何管理药物相互作用？",
                    "什么是药物相互作用的数据库？",
                    "如何识别药物相互作用的信号？",
                    "什么是药物相互作用的机制？",
                    "如何预防严重的药物相互作用？",
                    "什么是药物相互作用的时间效应？",
                    "如何评估药物相互作用的程度？",
                    "什么是药物相互作用的个体差异？",
                    "如何监测药物相互作用的效果？"
                ]
            }
            
            if question_data["category"] in variants_map:
                variants = variants_map[question_data["category"]]
                for variant in variants:
                    variant_sample = base_sample.copy()
                    variant_sample["id"] = f"pharma_{len(self.samples)}"
                    variant_sample["question"] = variant
                    variant_sample["source"] = "variant"
                    self.samples.append(variant_sample)
        
        logger.info(f"生成了 {len(self.samples)} 个示例样本")
    
    def _preprocess_data(self):
        """数据预处理和增强"""
        processed_samples = []
        
        for sample in self.samples:
            # 清理问题文本
            question = sample['question'].strip()
            if not question.endswith('？') and not question.endswith('?'):
                question += '？'
            
            # 添加难度权重
            difficulty_weights = {'easy': 1.0, 'medium': 1.2, 'hard': 1.5}
            weight = difficulty_weights.get(sample['difficulty'], 1.0)
            
            # 添加类别编码
            category_map = {
                '药物副作用': 0, '药物储存': 1, '用药原则': 2,
                '慢性病管理': 3, '儿科用药': 4, '药理学': 5,
                '药物安全': 6, '营养补充': 7, '精神药物': 8
            }
            category_id = category_map.get(sample['category'], -1)
            
            processed_sample = sample.copy()
            processed_sample.update({
                'question_processed': question,
                'weight': weight,
                'category_id': category_id,
                'length': len(question)
            })
            
            processed_samples.append(processed_sample)
        
        self.samples = processed_samples
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        if idx >= len(self.samples):
            raise IndexError(f"索引 {idx} 超出数据集范围")
        
        sample = self.samples[idx].copy()
        
        # 转换为模型输入格式
        model_input = {
            'input_ids': torch.tensor([hash(sample['question']) % 10000]),  # 模拟token ID
            'attention_mask': torch.ones(1),
            'labels': torch.tensor([sample.get('category_id', 0)]),
            'metadata': sample
        }
        
        return model_input
    
    def get_sample_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类别获取样本"""
        return [s for s in self.samples if s.get('category') == category]
    
    def get_sample_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """按难度获取样本"""
        return [s for s in self.samples if s.get('difficulty') == difficulty]
    
    def get_categories(self) -> List[str]:
        """获取所有类别"""
        return list(set(s.get('category', 'unknown') for s in self.samples))
    
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """获取难度分布"""
        distribution = {'easy': 0, 'medium': 0, 'hard': 0}
        for sample in self.samples:
            difficulty = sample.get('difficulty', 'easy')
            if difficulty in distribution:
                distribution[difficulty] += 1
        return distribution
    
    def get_category_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = {}
        for sample in self.samples:
            category = sample.get('category', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Tuple['PharmaKnowledgeDataset', 'PharmaKnowledgeDataset', 'PharmaKnowledgeDataset']:
        """分割数据集"""
        random.shuffle(self.samples)
        
        total_size = len(self.samples)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_samples = self.samples[:train_size]
        val_samples = self.samples[train_size:train_size + val_size]
        test_samples = self.samples[train_size + val_size:]
        
        train_dataset = PharmaKnowledgeDataset(data_list=train_samples, split='train')
        val_dataset = PharmaKnowledgeDataset(data_list=val_samples, split='val')
        test_dataset = PharmaKnowledgeDataset(data_list=test_samples, split='test')
        
        return train_dataset, val_dataset, test_dataset
    
    def search_samples(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索包含关键词的样本"""
        results = []
        keyword_lower = keyword.lower()
        
        for sample in self.samples:
            # 在问题中搜索
            if keyword_lower in sample.get('question', '').lower():
                results.append(sample)
                continue
            
            # 在关键词中搜索
            sample_keywords = sample.get('keywords', [])
            if any(keyword_lower in str(k).lower() for k in sample_keywords):
                results.append(sample)
                continue
        
        return results
    
    def save_dataset(self, file_path: str):
        """保存数据集到文件"""
        dataset_info = {
            'samples': self.samples,
            'metadata': {
                'total_samples': len(self.samples),
                'categories': self.get_categories(),
                'difficulty_distribution': self.get_difficulty_distribution(),
                'category_distribution': self.get_category_distribution(),
                'created_at': datetime.now().isoformat(),
                'split': self.split
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集已保存到: {file_path}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'total_samples': len(self.samples),
            'categories': self.get_categories(),
            'difficulty_distribution': self.get_difficulty_distribution(),
            'category_distribution': self.get_category_distribution(),
            'average_question_length': sum(len(s.get('question', '')) for s in self.samples) / len(self.samples),
            'split': self.split
        }