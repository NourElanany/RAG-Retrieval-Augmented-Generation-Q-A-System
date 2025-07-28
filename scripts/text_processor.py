import re
import string
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import ISRIStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from textblob import TextBlob

class ArabicTextProcessor:
    def __init__(self):
        """تهيئة معالج النصوص العربية"""
        # تحميل الموارد المطلوبة
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except:
            pass
        
        # تهيئة أدوات المعالجة
        self.stemmer = ISRIStemmer()
        self.arabic_stopwords = set([
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'التي', 'الذي', 'التي', 'اللذان', 'اللتان', 'اللذين', 'اللتين',
            'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'نحن',
            'كان', 'كانت', 'كانوا', 'كن', 'يكون', 'تكون', 'يكونوا', 'تكن',
            'قد', 'لقد', 'قال', 'قالت', 'قالوا', 'قلن', 'يقول', 'تقول',
            'أن', 'إن', 'كي', 'لكي', 'حتى', 'لو', 'إذا', 'إذ', 'بعد', 'قبل',
            'أم', 'أو', 'لكن', 'لكن', 'غير', 'سوى', 'عدا', 'خلا', 'حاشا'
        ])
        
        # تحميل نموذج spaCy للعربية (إذا كان متوفراً)
        try:
            self.nlp = spacy.load("ar_core_news_sm")
        except:
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """تنظيف النص من الرموز والأحرف غير المرغوبة"""
        if not text:
            return ""
        
        # إزالة الأرقام الإنجليزية والعربية
        text = re.sub(r'[0-9٠-٩]+', '', text)
        
        # إزالة علامات الترقيم
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)
        text = re.sub(r'[،؛؟!""''()\\[\\]{}]', ' ', text)
        
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text)
        
        # تطبيع الأحرف العربية
        text = self.normalize_arabic(text)
        
        return text.strip()
    
    def normalize_arabic(self, text: str) -> str:
        """تطبيع الأحرف العربية"""
        # تطبيع الألف
        text = re.sub(r'[إأآا]', 'ا', text)
        
        # تطبيع التاء المربوطة
        text = re.sub(r'ة', 'ه', text)
        
        # تطبيع الياء
        text = re.sub(r'ى', 'ي', text)
        
        # إزالة التشكيل
        text = re.sub(r'[ًٌٍَُِّْ]', '', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """تقسيم النص إلى كلمات"""
        try:
            if self.nlp:
                doc = self.nlp(text)
                return [token.text for token in doc if not token.is_space]
            else:
                return word_tokenize(text)
        except:
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """إزالة كلمات الوقف"""
        return [token for token in tokens if token not in self.arabic_stopwords]
    
    def stem_words(self, tokens: List[str]) -> List[str]:
        """استخراج جذور الكلمات"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def extract_entities(self, text: str) -> List[Dict]:
        """استخراج الكيانات المسماة"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return entities
    
    def get_pos_tags(self, tokens: List[str]) -> List[tuple]:
        """تحديد أجزاء الكلام"""
        try:
            return pos_tag(tokens)
        except:
            return [(token, 'UNKNOWN') for token in tokens]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """حساب التشابه بين نصين"""
        try:
            blob1 = TextBlob(text1)
            blob2 = TextBlob(text2)
            
            # حساب التشابه باستخدام Jaccard similarity
            set1 = set(blob1.words)
            set2 = set(blob2.words)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def process_text(self, text: str, full_processing: bool = True) -> Dict:
        """معالجة شاملة للنص"""
        result = {
            'original': text,
            'cleaned': self.clean_text(text),
            'tokens': [],
            'filtered_tokens': [],
            'stemmed_tokens': [],
            'entities': [],
            'pos_tags': []
        }
        
        if full_processing and result['cleaned']:
            result['tokens'] = self.tokenize(result['cleaned'])
            result['filtered_tokens'] = self.remove_stopwords(result['tokens'])
            result['stemmed_tokens'] = self.stem_words(result['filtered_tokens'])
            result['entities'] = self.extract_entities(text)
            result['pos_tags'] = self.get_pos_tags(result['tokens'])
        
        return result