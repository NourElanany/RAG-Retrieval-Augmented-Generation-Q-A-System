import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class ContextRetriever:
    def __init__(self, index_path, contexts_path, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # تحميل نموذج التمثيل الرقمي
        self.model = SentenceTransformer(model_name)
        
        # تحميل فهرس FAISS
        self.index = faiss.read_index(index_path)
        
        # تحميل السياقات
        with open(contexts_path, 'r', encoding='utf-8') as f:
            self.contexts = [line.strip() for line in f.readlines()]
    
    def retrieve(self, query, top_k=3):
        """استرجاع أفضل k سياقات ذات صلة بالاستعلام"""
        # تحويل الاستعلام إلى تمثيل رقمي
        query_embedding = self.model.encode([query])
        
        # تطبيع المتجه
        faiss.normalize_L2(query_embedding)
        
        # البحث في الفهرس
        scores, indices = self.index.search(query_embedding, top_k)
        
        # استرجاع السياقات المقابلة
        retrieved_contexts = [self.contexts[idx] for idx in indices[0]]
        retrieved_scores = scores[0].tolist()
        
        return list(zip(retrieved_contexts, retrieved_scores))

def main():
    # مسارات الملفات
    embeddings_dir = "../embeddings"
    index_path = os.path.join(embeddings_dir, "faiss_index.index")
    contexts_path = os.path.join(embeddings_dir, "unique_contexts.txt")
    
    # التحقق من وجود الملفات
    if not os.path.exists(index_path) or not os.path.exists(contexts_path):
        print("لم يتم العثور على فهرس FAISS أو ملف السياقات. قم بتشغيل generate_embeddings.py و build_index.py أولاً.")
        return
    
    # إنشاء مسترجع السياقات
    retriever = ContextRetriever(index_path, contexts_path)
    
    # اختبار الاسترجاع
    query = input("أدخل سؤالك: ")
    results = retriever.retrieve(query)
    
    print("\nالسياقات ذات الصلة:")
    for i, (context, score) in enumerate(results):
        print(f"\n{i+1}. درجة التشابه: {score:.4f}")
        print(f"السياق: {context}")

if __name__ == "__main__":
    main()