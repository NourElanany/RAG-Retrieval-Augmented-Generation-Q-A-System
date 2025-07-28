import os
import numpy as np
import faiss

def build_faiss_index(embeddings, index_path):
    """بناء فهرس FAISS للبحث السريع"""
    # الحصول على أبعاد التمثيلات الرقمية
    dimension = embeddings.shape[1]
    
    # إنشاء فهرس FAISS
    index = faiss.IndexFlatIP(dimension)  # استخدام Inner Product للتشابه
    
    # إضافة التمثيلات الرقمية إلى الفهرس
    faiss.normalize_L2(embeddings)  # تطبيع المتجهات للحصول على تشابه الجيب تمام
    index.add(embeddings)
    
    # حفظ الفهرس
    faiss.write_index(index, index_path)
    print(f"تم حفظ فهرس FAISS في {index_path}")
    return index

def main():
    # التأكد من وجود مجلد التمثيلات الرقمية
    embeddings_dir = "embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # تحميل التمثيلات الرقمية
    embeddings_path = os.path.join(embeddings_dir, "context_embeddings.npy")
    if not os.path.exists(embeddings_path):
        print("لم يتم العثور على ملف التمثيلات الرقمية. قم بتشغيل generate_embeddings.py أولاً.")
        return
    
    embeddings = np.load(embeddings_path)
    print(f"تم تحميل {embeddings.shape[0]} تمثيل رقمي بأبعاد {embeddings.shape[1]}")
    
    # بناء وحفظ فهرس FAISS
    index_path = os.path.join(embeddings_dir, "faiss_index.index")
    build_faiss_index(embeddings, index_path)

if __name__ == "__main__":
    main()