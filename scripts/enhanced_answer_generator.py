from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from text_processor import ArabicTextProcessor
from typing import List, Dict
import re

class EnhancedAnswerGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """تهيئة مولد الإجابات المحسن"""
        self.text_processor = ArabicTextProcessor()
        
        print(f"تحميل نموذج {model_name}...")
        try:
            # استخدام نموذج أكثر استقراراً
            self.generator = pipeline(
                "text2text-generation",
                model="t5-small",
                tokenizer="t5-small",
                max_length=512,
                device=-1
            )
        except Exception as e:
            print(f"خطأ في تحميل النموذج: {e}")
            # نموذج احتياطي
            self.generator = pipeline(
                "text-generation",
                model="gpt2",
                max_length=200,
                device=-1
            )
    
    def extract_answer_from_context(self, question: str, context: str) -> str:
        """استخراج الإجابة من السياق باستخدام قواعد NLP"""
        # معالجة السؤال والسياق
        question_analysis = self.text_processor.process_text(question)
        context_analysis = self.text_processor.process_text(context)
        
        # البحث عن الكيانات المشتركة
        question_entities = [ent['text'] for ent in question_analysis['entities']]
        context_entities = [ent['text'] for ent in context_analysis['entities']]
        
        # البحث عن الجمل ذات الصلة
        context_sentences = context.split('.')
        relevant_sentences = []
        
        for sentence in context_sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) > 10:
                # حساب التشابه مع السؤال
                similarity = self.text_processor.calculate_similarity(question, sentence_clean)
                if similarity > 0.1:
                    relevant_sentences.append((sentence_clean, similarity))
        
        # ترتيب الجمل حسب الصلة
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            return relevant_sentences[0][0]
        
        return ""
    
    def validate_answer(self, question: str, answer: str, context: str) -> Dict:
        """التحقق من صحة الإجابة"""
        validation = {
            'is_valid': True,
            'confidence': 0.0,
            'issues': []
        }
        
        # فحص طول الإجابة
        if len(answer.strip()) < 5:
            validation['is_valid'] = False
            validation['issues'].append('الإجابة قصيرة جداً')
        
        # فحص التشابه مع السياق
        context_similarity = self.text_processor.calculate_similarity(answer, context)
        if context_similarity < 0.1:
            validation['issues'].append('الإجابة لا تتطابق مع السياق')
            validation['confidence'] -= 0.3
        
        # فحص التشابه مع السؤال
        question_similarity = self.text_processor.calculate_similarity(answer, question)
        if question_similarity > 0.8:
            validation['issues'].append('الإجابة مطابقة للسؤال')
            validation['confidence'] -= 0.2
        
        # حساب الثقة النهائية
        validation['confidence'] = max(0.0, min(1.0, context_similarity + 0.5))
        
        return validation
    
    def generate_answer(self, question: str, contexts: List) -> Dict:
        """توليد إجابة محسنة بناءً على السؤال والسياقات"""
        try:
            # استخراج النصوص من tuples إذا لزم الأمر
            if contexts and isinstance(contexts[0], tuple):
                context_texts = [context[0] for context in contexts[:3]]
                context_scores = [context[1] for context in contexts[:3]]
            elif contexts and isinstance(contexts[0], dict):
                context_texts = [context['context'] for context in contexts[:3]]
                context_scores = [context['final_score'] for context in contexts[:3]]
            else:
                context_texts = contexts[:3]
                context_scores = [1.0] * len(context_texts)
            
            # محاولة استخراج إجابة مباشرة من السياق
            direct_answers = []
            for context in context_texts:
                direct_answer = self.extract_answer_from_context(question, context)
                if direct_answer:
                    direct_answers.append(direct_answer)
            
            # تجميع السياقات
            context_text = "\n".join(context_texts)
            
            # تكوين النص المدخل
            if "t5" in str(self.generator.model.config._name_or_path).lower():
                input_text = f"question: {question} context: {context_text}"
            else:
                input_text = f"السياق: {context_text}\nالسؤال: {question}\nالإجابة:"
            
            # توليد الإجابة
            result = self.generator(
                input_text,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            generated_answer = ""
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    generated_answer = result[0]['generated_text']
                    if input_text in generated_answer:
                        generated_answer = generated_answer.replace(input_text, "").strip()
                elif 'text' in result[0]:
                    generated_answer = result[0]['text']
            
            # اختيار أفضل إجابة
            candidates = []
            
            if direct_answers:
                candidates.extend(direct_answers)
            
            if generated_answer:
                candidates.append(generated_answer)
            
            # تقييم الإجابات المرشحة
            best_answer = "عذراً، لم أتمكن من العثور على إجابة مناسبة."
            best_validation = {'confidence': 0.0}
            
            for candidate in candidates:
                validation = self.validate_answer(question, candidate, context_text)
                if validation['confidence'] > best_validation['confidence']:
                    best_answer = candidate
                    best_validation = validation
            
            return {
                'answer': best_answer,
                'confidence': best_validation['confidence'],
                'validation': best_validation,
                'context_scores': context_scores,
                'used_contexts': len(context_texts)
            }
            
        except Exception as e:
            print(f"خطأ في توليد الإجابة: {e}")
            return {
                'answer': f"حدث خطأ أثناء توليد الإجابة: {str(e)}",
                'confidence': 0.0,
                'validation': {'is_valid': False, 'issues': [str(e)]},
                'context_scores': [],
                'used_contexts': 0
            }