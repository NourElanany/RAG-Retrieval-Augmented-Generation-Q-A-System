import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from text_processor import ArabicTextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedContextRetriever:
    def __init__(self, index_path, contexts_path, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # تحميل معالج النصوص
        self.text_processor = ArabicTextProcessor()
        
        # تحميل نموذج التمثيل الرقمي
        self.model = SentenceTransformer(model_name)
        
        # تحميل فهرس FAISS
        self.index = faiss.read_index(index_path)
        
        # تحميل السياقات
        with open(contexts_path, 'r', encoding='utf-8') as f:
            self.contexts = [line.strip() for line in f.readlines()]
        
        # إنشاء TF-IDF vectorizer للبحث التقليدي
        self.setup_tfidf()
    
    def setup_tfidf(self):
        """إعداد TF-IDF للبحث التقليدي"""
        processed_contexts = []
        for context in self.contexts:
            processed = self.text_processor.process_text(context)
            processed_text = ' '.join(processed['stemmed_tokens'])
            processed_contexts.append(processed_text)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_contexts)
        except:
            self.tfidf_matrix = None
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """البحث الدلالي باستخدام FAISS"""
        # معالجة الاستعلام
        processed_query = self.text_processor.process_text(query)
        
        # تحويل الاستعلام إلى تمثيل رقمي
        query_embedding = self.model.encode([processed_query['cleaned']])
        
        # تطبيع المتجه
        faiss.normalize_L2(query_embedding)
        
        # البحث في الفهرس
        scores, indices = self.index.search(query_embedding, top_k)
        
        # استرجاع السياقات المقابلة
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.contexts):
                results.append((self.contexts[idx], float(score)))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """البحث بالكلمات المفتاحية باستخدام TF-IDF"""
        if self.tfidf_matrix is None:
            return []
        
        # معالجة الاستعلام
        processed_query = self.text_processor.process_text(query)
        processed_text = ' '.join(processed_query['stemmed_tokens'])
        
        try:
            # تحويل الاستعلام إلى متجه TF-IDF
            query_vector = self.tfidf_vectorizer.transform([processed_text])
            
            # حساب التشابه
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # ترتيب النتائج
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append((self.contexts[idx], float(similarities[idx])))
            
            return results
        except:
            return []
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """البحث المختلط (دلالي + كلمات مفتاحية)"""
        # البحث الدلالي
        semantic_results = self.semantic_search(query, top_k * 2)
        
        # البحث بالكلمات المفتاحية
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # دمج النتائج وإزالة التكرار
        combined_results = {}
        
        # إضافة النتائج الدلالية بوزن أعلى
        for context, score in semantic_results:
            combined_results[context] = score * 0.7
        
        # إضافة نتائج الكلمات المفتاحية
        for context, score in keyword_results:
            if context in combined_results:
                combined_results[context] += score * 0.3
            else:
                combined_results[context] = score * 0.3
        
        # ترتيب النتائج النهائية
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]
    
    def retrieve_with_context_analysis(self, query: str, top_k: int = 3) -> Dict:
        """استرجاع متقدم مع تحليل السياق"""
        # معالجة الاستعلام
        query_analysis = self.text_processor.process_text(query)
        
        # البحث المختلط
        results = self.hybrid_search(query, top_k)
        
        # تحليل النتائج
        analyzed_results = []
        for context, score in results:
            context_analysis = self.text_processor.process_text(context)
            
            # حساب التشابه النصي
            text_similarity = self.text_processor.calculate_similarity(
                query_analysis['cleaned'], 
                context_analysis['cleaned']
            )
            
            # تحليل الكيانات المشتركة
            query_entities = set([ent['text'] for ent in query_analysis['entities']])
            context_entities = set([ent['text'] for ent in context_analysis['entities']])
            entity_overlap = len(query_entities.intersection(context_entities))
            
            # حساب درجة التقييم النهائية
            final_score = (score + text_similarity + (entity_overlap * 0.2)) / 3
            
            analyzed_results.append({
                'context': context,
                'semantic_score': score,
                'text_similarity': text_similarity,
                'entity_overlap': entity_overlap,
                'final_score': final_score,
                'text_similarity': text_similarity,
                'entity_overlap': entity_overlap,
                'final_score': score * 0.6 + text_similarity * 0.3 + (entity_overlap * 0.1),
                'entities': context_analysis['entities']
            })
        
        # إعادة ترتيب حسب النتيجة النهائية
        analyzed_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            'analyzed_results': analyzed_results[:top_k],
            'query_analysis': query_analysis
        }
        analyzed_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            'query_analysis': query_analysis,
            'results': analyzed_results[:top_k]
        }