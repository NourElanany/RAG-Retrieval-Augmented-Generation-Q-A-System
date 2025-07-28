from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from advanced_text_processor import AdvancedArabicProcessor
from typing import List, Dict, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SmartAnswerGenerator:
    def __init__(self):
        """مولد إجابات ذكي متقدم"""
        self.text_processor = AdvancedArabicProcessor()
        
        # تحميل نماذج متعددة للحصول على أفضل النتائج
        self.models = {}
        
        try:
            # نموذج T5 للنصوص العربية
            self.models['t5'] = pipeline(
                "text2text-generation",
                model="t5-small",
                tokenizer="t5-small",
                max_length=512,
                device=-1
            )
        except Exception as e:
            print(f"خطأ في تحميل T5: {e}")
        
        try:
            # نموذج GPT للنصوص
            self.models['gpt'] = pipeline(
                "text-generation",
                model="gpt2",
                max_length=200,
                device=-1
            )
        except Exception as e:
            print(f"خطأ في تحميل GPT: {e}")
    
    def validate_answer_advanced(self, question: str, answer: str, contexts: List[str]) -> Dict:
        """تحقق متقدم من صحة الإجابة"""
        validation = {
            'is_valid': True,
            'confidence_score': 0.0,
            'quality_metrics': {},
            'issues': [],
            'strengths': []
        }
        
        # 1. فحص طول الإجابة
        answer_length = len(answer.strip())
        if answer_length < 5:
            validation['is_valid'] = False
            validation['issues'].append('الإجابة قصيرة جداً')
        elif answer_length > 500:
            validation['issues'].append('الإجابة طويلة جداً')
        else:
            validation['strengths'].append('طول الإجابة مناسب')
        
        # 2. فحص التشابه مع السياقات
        context_similarities = []
        for context in contexts:
            sim = self.text_processor.calculate_advanced_similarity(answer, context)
            context_similarities.append(sim['composite'])
        
        max_context_sim = max(context_similarities) if context_similarities else 0
        avg_context_sim = np.mean(context_similarities) if context_similarities else 0
        
        validation['quality_metrics']['max_context_similarity'] = max_context_sim
        validation['quality_metrics']['avg_context_similarity'] = avg_context_sim
        
        if max_context_sim < 0.1:
            validation['issues'].append('الإجابة لا تتطابق مع السياق')
        elif max_context_sim > 0.5:
            validation['strengths'].append('الإجابة مرتبطة بقوة بالسياق')
        
        # 3. فحص التشابه مع السؤال
        question_sim = self.text_processor.calculate_advanced_similarity(answer, question)
        validation['quality_metrics']['question_similarity'] = question_sim['composite']
        
        if question_sim['composite'] > 0.8:
            validation['issues'].append('الإجابة مطابقة للسؤال (تكرار)')
        elif question_sim['composite'] < 0.1:
            validation['issues'].append('الإجابة غير مرتبطة بالسؤال')
        else:
            validation['strengths'].append('الإجابة مرتبطة بالسؤال بشكل مناسب')
        
        # 4. فحص وجود معلومات جديدة
        question_info = self.text_processor.extract_question_type(question)
        answer_tokens = set(self.text_processor.advanced_clean_text(answer).split())
        question_tokens = set(self.text_processor.advanced_clean_text(question).split())
        
        new_info_ratio = len(answer_tokens - question_tokens) / len(answer_tokens) if answer_tokens else 0
        validation['quality_metrics']['new_information_ratio'] = new_info_ratio
        
        if new_info_ratio > 0.7:
            validation['strengths'].append('الإجابة تحتوي على معلومات جديدة')
        elif new_info_ratio < 0.3:
            validation['issues'].append('الإجابة تكرر السؤال بدون إضافة معلومات')
        
        # 5. فحص الكيانات والكلمات المفتاحية
        entity_coverage = 0
        for entity in question_info['entities']:
            if entity['text'] in answer:
                entity_coverage += 1
        
        if question_info['entities']:
            entity_coverage = entity_coverage / len(question_info['entities'])
            validation['quality_metrics']['entity_coverage'] = entity_coverage
            
            if entity_coverage > 0.5:
                validation['strengths'].append('الإجابة تغطي الكيانات المطلوبة')
        
        # حساب نتيجة الثقة المركبة
        confidence_factors = [
            max_context_sim * 0.4,  # التشابه مع السياق
            min(question_sim['composite'], 0.5) * 0.2,  # التشابه المعتدل مع السؤال
            new_info_ratio * 0.3,  # نسبة المعلومات الجديدة
            entity_coverage * 0.1 if question_info['entities'] else 0.1  # تغطية الكيانات
        ]
        
        validation['confidence_score'] = sum(confidence_factors)
        
        # تعديل الثقة بناءً على المشاكل
        if validation['issues']:
            validation['confidence_score'] *= (1 - len(validation['issues']) * 0.1)
        
        # تعزيز الثقة بناءً على نقاط القوة
        if validation['strengths']:
            validation['confidence_score'] *= (1 + len(validation['strengths']) * 0.05)
        
        validation['confidence_score'] = max(0.0, min(1.0, validation['confidence_score']))
        
        return validation
    
    def generate_smart_answer(self, question: str, contexts: List[Dict]) -> Dict:
        """توليد إجابة ذكية متقدمة"""
        try:
            # استخراج النصوص والنتائج
            if isinstance(contexts[0], dict):
                context_texts = [ctx['context'] for ctx in contexts[:3]]
                context_scores = [ctx.get('final_score', ctx.get('semantic_score', 1.0)) for ctx in contexts[:3]]
            else:
                context_texts = [str(ctx) for ctx in contexts[:3]]
                context_scores = [1.0] * len(context_texts)
            
            # تحليل السؤال
            question_info = self.text_processor.extract_question_type(question)
            
            # استخراج مرشحي الإجابات من كل سياق
            all_candidates = []
            for context in context_texts:
                candidates = self.text_processor.extract_answer_candidates(question, context)
                all_candidates.extend(candidates)
            
            # ترتيب جميع المرشحين
            all_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # اختيار أفضل المرشحين
            best_candidates = all_candidates[:3]
            
            # توليد إجابات باستخدام النماذج
            generated_answers = []
            
            # دمج السياقات
            combined_context = "\n".join(context_texts)
            
            # توليد باستخدام T5
            if 't5' in self.models:
                try:
                    input_text = f"question: {question} context: {combined_context}"
                    result = self.models['t5'](
                        input_text,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True
                    )
                    if result and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        if generated_text:
                            generated_answers.append({
                                'text': generated_text,
                                'source': 't5_generated',
                                'method': 'neural_generation'
                            })
                except Exception as e:
                    print(f"خطأ في T5: {e}")
            
            # إضافة المرشحين المستخرجين
            for candidate in best_candidates:
                generated_answers.append({
                    'text': candidate['text'],
                    'source': 'extracted',
                    'method': 'rule_based_extraction',
                    'score': candidate['composite_score']
                })
            
            # تقييم جميع الإجابات المرشحة
            evaluated_answers = []
            for answer_data in generated_answers:
                validation = self.validate_answer_advanced(
                    question, 
                    answer_data['text'], 
                    context_texts
                )
                
                evaluated_answers.append({
                    'text': answer_data['text'],
                    'validation': validation,
                    'source': answer_data['source'],
                    'method': answer_data['method'],
                    'final_score': validation['confidence_score']
                })
            
            # اختيار أفضل إجابة
            if evaluated_answers:
                best_answer = max(evaluated_answers, key=lambda x: x['final_score'])
                
                # إذا كانت أفضل إجابة ضعيفة، نحاول تحسينها
                if best_answer['final_score'] < 0.3:
                    # إنشاء إجابة مركبة من أفضل المرشحين
                    top_candidates = [ans for ans in evaluated_answers if ans['final_score'] > 0.1]
                    if len(top_candidates) > 1:
                        combined_answer = self.combine_answers(top_candidates[:2])
                        combined_validation = self.validate_answer_advanced(
                            question, combined_answer, context_texts
                        )
                        
                        if combined_validation['confidence_score'] > best_answer['final_score']:
                            best_answer = {
                                'text': combined_answer,
                                'validation': combined_validation,
                                'source': 'combined',
                                'method': 'intelligent_combination',
                                'final_score': combined_validation['confidence_score']
                            }
            else:
                best_answer = {
                    'text': 'عذراً، لم أتمكن من العثور على إجابة مناسبة في السياق المتاح.',
                    'validation': {'confidence_score': 0.0, 'issues': ['لا توجد إجابة مناسبة']},
                    'source': 'fallback',
                    'method': 'fallback',
                    'final_score': 0.0
                }
            
            return {
                'answer': best_answer['text'],
                'confidence': best_answer['final_score'],
                'validation': best_answer['validation'],
                'source': best_answer['source'],
                'method': best_answer['method'],
                'context_scores': context_scores,
                'used_contexts': len(context_texts),
                'question_analysis': question_info,
                'all_candidates': len(evaluated_answers)
            }
            
        except Exception as e:
            return {
                'answer': f'حدث خطأ في النظام: {str(e)}',
                'confidence': 0.0,
                'validation': {'issues': [str(e)]},
                'source': 'error',
                'method': 'error_handling',
                'context_scores': [],
                'used_contexts': 0
            }
    
    def combine_answers(self, candidates: List[Dict]) -> str:
        """دمج إجابات متعددة لإنشاء إجابة محسنة"""
        if not candidates:
            return ""
        
        if len(candidates) == 1:
            return candidates[0]['text']
        
        # استخراج الجمل الأكثر أهمية من كل مرشح
        important_sentences = []
        
        for candidate in candidates:
            sentences = candidate['text'].split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    important_sentences.append(sentence)
        
        # إزالة التكرار
        unique_sentences = []
        for sentence in important_sentences:
            is_duplicate = False
            for existing in unique_sentences:
                similarity = self.text_processor.calculate_advanced_similarity(sentence, existing)
                if similarity['composite'] > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        # دمج الجمل
        combined = '. '.join(unique_sentences[:3])  # أفضل 3 جمل
        return combined + '.' if combined and not combined.endswith('.') else combined