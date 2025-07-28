import re
import string
import os
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import ISRIStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob
import difflib

class AdvancedArabicProcessor:
    def __init__(self):
        """معالج نصوص عربي متقدم مع ذكاء اصطناعي"""
            # تعيين مسار تنزيل NLTK داخل المشروع
        self.nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
        
        # تهيئة حالة NLTK
        self.nltk_available = False
        self.punkt_available = False
        
        try:
            # إنشاء مجلد NLTK_DATA إذا لم يكن موجوداً
            os.makedirs(self.nltk_data_dir, exist_ok=True)
            
            # إضافة مسار NLTK المخصص إلى المسارات
            if self.nltk_data_dir not in nltk.data.path:
                nltk.data.path.append(self.nltk_data_dir)
            
            # محاولة تحميل Punkt بدون تنزيل
            punkt_path = os.path.join(self.nltk_data_dir, 'tokenizers', 'punkt')
            if os.path.exists(punkt_path):
                self.punkt_available = True
            
            # محاولة تنزيل Punkt إذا لم يكن موجوداً
            if not self.punkt_available:
                try:
                    nltk.download('punkt', download_dir=self.nltk_data_dir, quiet=True)
                    self.punkt_available = True
                except Exception as e:
                    print(f"تحذير: تعذر تنزيل Punkt tokenizer - سيتم استخدام البديل البسيط: {str(e)}")
            
            # تنزيل الموارد الإضافية بشكل صامت
            optional_resources = ['stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
            for resource in optional_resources:
                resource_path = os.path.join(self.nltk_data_dir, resource)
                if not os.path.exists(resource_path):
                    try:
                        nltk.download(resource, download_dir=self.nltk_data_dir, quiet=True)
                    except Exception as e:
                        print(f"تحذير: تعذر تنزيل المورد {resource}: {str(e)}")
            
            self.nltk_available = True
        
        except Exception as e:
            print(f"تحذير: خطأ في إعداد موارد NLTK - سيتم استخدام التحليل البسيط: {str(e)}")        # أدوات متقدمة
        self.stemmer = ISRIStemmer()
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # كلمات الوقف المحسنة
        self.arabic_stopwords = set([
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'التي', 'الذي', 'اللذان', 'اللتان', 'اللذين', 'اللتين',
            'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'نحن',
            'كان', 'كانت', 'كانوا', 'كن', 'يكون', 'تكون', 'يكونوا', 'تكن',
            'قد', 'لقد', 'قال', 'قالت', 'قالوا', 'قلن', 'يقول', 'تقول',
            'أن', 'إن', 'كي', 'لكي', 'حتى', 'لو', 'إذا', 'إذ', 'بعد', 'قبل',
            'أم', 'أو', 'لكن', 'غير', 'سوى', 'عدا', 'خلا', 'حاشا', 'ليس',
            'ما', 'لا', 'لم', 'لن', 'كل', 'بعض', 'جميع', 'كلا', 'كلتا'
        ])
        
        # أنماط الأسئلة العربية
        self.question_patterns = {
            'من': ['شخص', 'اسم', 'هوية'],
            'ما': ['شيء', 'تعريف', 'وصف'],
            'متى': ['وقت', 'تاريخ', 'زمن'],
            'أين': ['مكان', 'موقع', 'جغرافيا'],
            'كيف': ['طريقة', 'كيفية', 'أسلوب'],
            'لماذا': ['سبب', 'علة', 'تفسير'],
            'كم': ['عدد', 'كمية', 'مقدار']
        }
        
        # تحميل spaCy
        try:
            self.nlp = spacy.load("ar_core_news_sm")
        except:
            self.nlp = None
    
    def advanced_clean_text(self, text: str) -> str:
        """تنظيف متقدم للنص"""
        if not text:
            return ""
        
        # إزالة HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # إزالة URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # إزالة الأرقام الإنجليزية والعربية
        text = re.sub(r'[0-9٠-٩]+', ' ', text)
        
        # إزالة علامات الترقيم مع الحفاظ على النقاط المهمة
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)
        text = re.sub(r'[،؛؟!""''()\\[\\]{}]', ' ', text)
        
        # تطبيع المسافات
        text = re.sub(r'\s+', ' ', text)
        
        # تطبيع الأحرف العربية المتقدم
        text = self.advanced_normalize_arabic(text)
        
        return text.strip()
    
    def advanced_normalize_arabic(self, text: str) -> str:
        """تطبيع متقدم للأحرف العربية"""
        # تطبيع الألف
        text = re.sub(r'[إأآا]', 'ا', text)
        
        # تطبيع التاء المربوطة والهاء
        text = re.sub(r'ة', 'ه', text)
        
        # تطبيع الياء
        text = re.sub(r'ى', 'ي', text)
        
        # إزالة التشكيل
        text = re.sub(r'[ًٌٍَُِّْ]', '', text)
        
        # تطبيع الواو
        text = re.sub(r'ؤ', 'و', text)
        
        # تطبيع الياء المهموزة
        text = re.sub(r'ئ', 'ي', text)
        
        return text
    
    def simple_word_tokenize(self, text: str) -> List[str]:
        """تقسيم بسيط للنص إلى كلمات"""
        if not text:
            return []
        
        try:
            # تنظيف النص أولاً
            text = self.advanced_clean_text(text)
            
            # إزالة علامات الترقيم الإضافية
            text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
            text = re.sub(r'_', ' ', text)  # معالجة الشرطة السفلية
            text = re.sub(r'\s+', ' ', text)  # تطبيع المسافات
            
            # تقسيم على المسافات وتنظيف
            words = [word.strip() for word in text.split()]
            
            # تصفية الكلمات
            words = [w for w in words if len(w) > 0 and any(c.isalpha() for c in w)]
            
            return words
            
        except Exception as e:
            print(f"تحذير: خطأ في التقسيم البسيط للكلمات: {str(e)}")
            # إرجاع النص المقسم على المسافات كحل بديل نهائي
            return [w for w in text.split() if w.strip()]
    
    def extract_question_type(self, question: str) -> Dict:
        """تحديد نوع السؤال وما يتوقع في الإجابة"""
        # تهيئة معلومات السؤال الافتراضية
        question_info = {
            'type': 'عام',
            'expected_answer_type': 'معلومات عامة',
            'keywords': [],
            'entities': [],
            'strengths': [],
            'issues': []
        }
        
        try:
            # معالجة النص وتجهيزه
            question_clean = self.advanced_clean_text(question)
            
            # محاولة استخدام NLTK للتقسيم إلى كلمات بصمت في حالة الفشل
            tokens = []
            if self.nltk_available and self.punkt_available:
                try:
                    tokens = word_tokenize(question_clean)
                except:
                    # استخدام التقسيم البسيط بصمت
                    pass
            
            # استخدام التقسيم البسيط إذا لم تنجح NLTK
            if not tokens:
                tokens = self.simple_word_tokenize(question_clean)
            
            # استخراج الكلمات المفتاحية
            filtered_tokens = [token for token in tokens if token not in self.arabic_stopwords and len(token) > 2]
            question_info['keywords'] = filtered_tokens if filtered_tokens else []
            if filtered_tokens:
                question_info['strengths'].append(f'تم العثور على {len(filtered_tokens)} كلمة مفتاحية')
            
            # تحديد نوع السؤال من الأنماط المعروفة
            for pattern, answer_types in self.question_patterns.items():
                if pattern in tokens:
                    question_info['type'] = pattern
                    question_info['expected_answer_type'] = answer_types[0]
                    question_info['strengths'].append(f'نوع السؤال معروف: {pattern}')
                    break
            
            # محاولة تحديد نوع السؤال بطريقة بديلة
            if question_info['type'] == 'عام':
                question_words = ['من', 'ماذا', 'متى', 'أين', 'كيف', 'لماذا']
                found_words = [word for word in question_words if word in tokens]
                if found_words:
                    question_info['type'] = 'استفهام'
                    question_info['expected_answer_type'] = 'معلومات محددة'
                    question_info['strengths'].append(f'تم تحديد كلمة استفهام: {found_words[0]}')
            
            # استخراج الكيانات باستخدام spaCy إذا كان متاحاً
            if self.nlp:
                try:
                    doc = self.nlp(question)
                    entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
                    question_info['entities'] = entities
                    if entities:
                        question_info['strengths'].append(f'تم العثور على {len(entities)} كيان محدد')
                except Exception as e:
                    question_info['issues'].append(f'خطأ في استخراج الكيانات: {str(e)}')
            
        except Exception as e:
            error_msg = f"خطأ في تحليل السؤال: {str(e)}"
            print(f"Warning: {error_msg}")
            question_info['issues'].append(error_msg)
        
        return question_info
    
    def calculate_advanced_similarity(self, text1: str, text2: str) -> Dict:
        """حساب التشابه المتقدم بين النصوص"""
        # تنظيف النصوص
        clean1 = self.advanced_clean_text(text1)
        clean2 = self.advanced_clean_text(text2)
        
        similarities = {}
        
        # 1. Jaccard Similarity with n-grams
        def get_ngrams(text, n):
            words = text.split()
            ngrams = []
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i+n]))
            return set(ngrams)
        
        # حساب تشابه جاكارد للكلمات المفردة و الثنائية والثلاثية
        unigrams_sim = 0
        bigrams_sim = 0
        trigrams_sim = 0
        
        # Unigrams
        set1_uni = set(clean1.split())
        set2_uni = set(clean2.split())
        inter_uni = len(set1_uni.intersection(set2_uni))
        union_uni = len(set1_uni.union(set2_uni))
        unigrams_sim = inter_uni / union_uni if union_uni > 0 else 0.0
        
        # Bigrams
        set1_bi = get_ngrams(clean1, 2)
        set2_bi = get_ngrams(clean2, 2)
        inter_bi = len(set1_bi.intersection(set2_bi))
        union_bi = len(set1_bi.union(set2_bi))
        bigrams_sim = inter_bi / union_bi if union_bi > 0 else 0.0
        
        # Trigrams
        set1_tri = get_ngrams(clean1, 3)
        set2_tri = get_ngrams(clean2, 3)
        inter_tri = len(set1_tri.intersection(set2_tri))
        union_tri = len(set1_tri.union(set2_tri))
        trigrams_sim = inter_tri / union_tri if union_tri > 0 else 0.0
        
        # Combined Jaccard similarity with weights
        similarities['jaccard'] = (unigrams_sim * 0.5 + bigrams_sim * 0.3 + trigrams_sim * 0.2)
        
        # 2. Cosine Similarity using TF-IDF with better error handling
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Include bigrams
            tfidf_matrix = vectorizer.fit_transform([clean1, clean2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarities['cosine_tfidf'] = float(cosine_sim)
        except Exception as e:
            print(f"تحذير في حساب تشابه TF-IDF: {str(e)}")
            similarities['cosine_tfidf'] = 0.0
        
        # 3. Semantic Similarity using Sentence Transformers with confidence
        try:
            embeddings = self.sentence_model.encode([clean1, clean2])
            semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            # تطبيق معامل ثقة للتشابه الدلالي
            confidence = min(len(clean1.split()), len(clean2.split())) / 20  # معامل ثقة بناءً على طول النص
            confidence = min(1.0, max(0.5, confidence))  # تقييد معامل الثقة بين 0.5 و 1.0
            similarities['semantic'] = float(semantic_sim * confidence)
        except Exception as e:
            print(f"تحذير في حساب التشابه الدلالي: {str(e)}")
            similarities['semantic'] = 0.0
        
        # 4. Sequence Similarity
        similarities['sequence'] = difflib.SequenceMatcher(None, clean1, clean2).ratio()
        
        # حساب مقاييس إضافية للتشابه
        
        # 5. تشابه المحتوى المعلوماتي
        def information_density(text):
            words = set(text.split())
            total_chars = sum(len(word) for word in words)
            return len(words) / total_chars if total_chars > 0 else 0
        
        info_sim1 = information_density(clean1)
        info_sim2 = information_density(clean2)
        similarities['info_density'] = 1 - abs(info_sim1 - info_sim2)
        
        # 6. تقييم جودة الإجابة
        def answer_quality(text):
            words = text.split()
            # عوامل الجودة: طول مناسب، وجود كلمات مفتاحية محددة
            length_score = min(1.0, len(words) / 20)  # الطول المثالي حوالي 20 كلمة
            has_numbers = any(char.isdigit() for char in text)
            has_special_chars = any(char in '٪$@#' for char in text)
            format_score = (0.7 + (0.15 if has_numbers else 0) + (0.15 if has_special_chars else 0))
            return (length_score * 0.7 + format_score * 0.3)
        
        similarities['quality'] = answer_quality(clean2)
        
        # حساب التشابه المركب مع الأوزان المحسنة
        similarities['composite'] = (
            similarities['jaccard'] * 0.25 +      # تشابه المحتوى
            similarities['cosine_tfidf'] * 0.20 + # تشابه الكلمات
            similarities['semantic'] * 0.30 +     # التشابه الدلالي
            similarities['sequence'] * 0.10 +     # تشابه التسلسل
            similarities['info_density'] * 0.05 + # كثافة المعلومات
            similarities['quality'] * 0.10        # جودة الإجابة
        )
        
        # إضافة تصنيف الثقة
        if similarities['composite'] > 0.8:
            similarities['confidence'] = 'عالية جداً'
        elif similarities['composite'] > 0.6:
            similarities['confidence'] = 'عالية'
        elif similarities['composite'] > 0.4:
            similarities['confidence'] = 'متوسطة'
        elif similarities['composite'] > 0.2:
            similarities['confidence'] = 'منخفضة'
        else:
            similarities['confidence'] = 'منخفضة جداً'
        
        return similarities
    
    def simple_sentence_tokenize(self, text: str) -> List[str]:
        """تقسيم بسيط للنص إلى جمل"""
        if not text:
            return []
        
        try:
            # تقسيم النص على علامات الترقيم المعروفة
            sentence_delimiters = r'[.。．؟!?\n]{1,}|[\n]{2,}'
            potential_sentences = re.split(sentence_delimiters, text)
            
            # تنظيف وتصفية الجمل
            sentences = []
            for sentence in potential_sentences:
                # تنظيف الجملة
                clean_sentence = sentence.strip()
                
                # التحقق من طول وجودة الجملة
                if len(clean_sentence) > 5 and any(c.isalpha() for c in clean_sentence):
                    sentences.append(clean_sentence)
            
            # في حالة عدم وجود جمل، حاول التقسيم على الفواصل
            if not sentences and '،' in text:
                sentences = [s.strip() for s in text.split('،') if len(s.strip()) > 5]
            
            # إذا لم يتم العثور على جمل، أعد النص كاملاً
            if not sentences and len(text.strip()) > 0:
                sentences = [text.strip()]
                
            return sentences
            
        except Exception as e:
            print(f"تحذير: خطأ في التقسيم البسيط للجمل: {str(e)}")
            # إرجاع النص كاملاً كجملة واحدة كحل بديل نهائي
            return [text.strip()] if text.strip() else []

    def extract_answer_candidates(self, question: str, context: str) -> List[Dict]:
        """استخراج مرشحي الإجابات من السياق"""
        question_info = self.extract_question_type(question)
        
        # اختيار طريقة التقسيم المناسبة للجمل
        sentences = []
        if self.nltk_available and self.punkt_available:
            try:
                sentences = sent_tokenize(context)
            except:
                # استخدام التقسيم البسيط بصمت
                pass
        
        # استخدام التقسيم البسيط إذا لم تنجح محاولة NLTK
        if not sentences:
            sentences = self.simple_sentence_tokenize(context)
        
        candidates = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
            
            # حساب التشابه مع السؤال
            similarity = self.calculate_advanced_similarity(question, sentence)
            
            # فحص وجود الكلمات المفتاحية
            keyword_score = 0
            for keyword in question_info['keywords']:
                if keyword in sentence:
                    keyword_score += 1
            keyword_score = keyword_score / len(question_info['keywords']) if question_info['keywords'] else 0
            
            # فحص وجود الكيانات
            entity_score = 0
            for entity in question_info['entities']:
                if entity['text'] in sentence:
                    entity_score += 1
            entity_score = entity_score / len(question_info['entities']) if question_info['entities'] else 0
            
            # حساب النتيجة المركبة
            composite_score = (
                similarity['composite'] * 0.5 +
                keyword_score * 0.3 +
                entity_score * 0.2
            )
            
            candidates.append({
                'text': sentence.strip(),
                'position': i,
                'similarity': similarity,
                'keyword_score': keyword_score,
                'entity_score': entity_score,
                'composite_score': composite_score
            })
        
        # تطبيق فلترة إضافية وترتيب المرشحين
        filtered_candidates = []
        for candidate in candidates:
            if candidate['composite_score'] > 0.2:  # استبعاد الإجابات ذات الدرجة المنخفضة جداً
                # إضافة معلومات تفصيلية عن الإجابة
                candidate['details'] = {
                    'length': len(candidate['text'].split()),
                    'has_numbers': any(char.isdigit() for char in candidate['text']),
                    'has_entities': candidate['entity_score'] > 0,
                    'keyword_matches': round(candidate['keyword_score'] * 100, 1)
                }
                
                # تحديد مستوى الثقة
                if candidate['composite_score'] > 0.8:
                    candidate['confidence'] = 'عالية جداً'
                elif candidate['composite_score'] > 0.6:
                    candidate['confidence'] = 'عالية'
                elif candidate['composite_score'] > 0.4:
                    candidate['confidence'] = 'متوسطة'
                else:
                    candidate['confidence'] = 'منخفضة'
                
                # تحديد سبب الاختيار
                reasons = []
                if candidate['similarity']['semantic'] > 0.7:
                    reasons.append('تشابه دلالي قوي')
                if candidate['keyword_score'] > 0.5:
                    reasons.append('تطابق الكلمات المفتاحية')
                if candidate['entity_score'] > 0:
                    reasons.append('تطابق الكيانات')
                if len(candidate['text'].split()) >= 5 and len(candidate['text'].split()) <= 30:
                    reasons.append('طول مناسب')
                candidate['selection_reasons'] = reasons
                
                filtered_candidates.append(candidate)
        
        # ترتيب المرشحين
        filtered_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return filtered_candidates[:5]  # أفضل 5 مرشحين مع تفاصيلهم