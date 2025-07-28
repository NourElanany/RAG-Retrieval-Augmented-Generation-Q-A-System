from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class AnswerGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """تهيئة مولد الإجابات"""
        print(f"تحميل نموذج {model_name}...")
        try:
            # استخدام نموذج أكثر استقراراً
            self.generator = pipeline(
                "text2text-generation",
                model="t5-small",  # نموذج أبسط وأكثر استقراراً
                tokenizer="t5-small",
                max_length=512,
                device=-1  # استخدام CPU
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
    
    def generate_answer(self, question, contexts):
        """توليد إجابة بناءً على السؤال والسياقات المسترجعة"""
        try:
            # استخراج النصوص من tuples (context, score)
            if contexts and isinstance(contexts[0], tuple):
                context_texts = [context[0] for context in contexts[:3]]
            else:
                context_texts = contexts[:3]
            
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
            
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    answer = result[0]['generated_text']
                    # تنظيف الإجابة
                    if input_text in answer:
                        answer = answer.replace(input_text, "").strip()
                    return answer
                elif 'text' in result[0]:
                    return result[0]['text']
            
            return "عذراً، لم أتمكن من توليد إجابة مناسبة."
            
        except Exception as e:
            print(f"خطأ في توليد الإجابة: {e}")
            return f"حدث خطأ أثناء توليد الإجابة: {str(e)}"

def main():
    # إنشاء مولد الإجابات
    generator = AnswerGenerator()
    
    # اختبار توليد الإجابة
    question = input("أدخل سؤالك: ")
    context = input("أدخل السياق: ")
    
    # محاكاة نتائج الاسترجاع
    contexts = [(context, 0.95)]
    
    # توليد الإجابة
    answer = generator.generate_answer(question, contexts)
    print(f"\nالإجابة: {answer}")

if __name__ == "__main__":
    main()