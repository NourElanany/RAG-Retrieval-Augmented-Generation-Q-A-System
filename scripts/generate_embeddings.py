import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(file_path):
    """تحميل البيانات من ملف CSV"""
    df = pd.read_csv(file_path)
    return df

def generate_embeddings(contexts, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """تحويل السياقات إلى embeddings باستخدام نموذج متعدد اللغات"""
    print(f"تحميل نموذج {model_name}...")
    model = SentenceTransformer(model_name)
    print("توليد التمثيلات الرقمية للسياقات...")
    embeddings = model.encode(contexts, show_progress_bar=True)
    return embeddings

def main():
    # إنشاء مجلد للتمثيلات الرقمية إذا لم يكن موجودًا
    os.makedirs("embeddings", exist_ok=True)
    
    # تحميل البيانات
    train_df = load_data("data/train.csv")
    val_df = load_data("data/validation.csv")
    
    # دمج البيانات للحصول على جميع السياقات
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # إزالة السياقات المكررة
    unique_contexts = all_df['context'].unique()
    print(f"عدد السياقات الفريدة: {len(unique_contexts)}")
    
    # توليد التمثيلات الرقمية
    context_embeddings = generate_embeddings(unique_contexts)
    
    # حفظ التمثيلات الرقمية والسياقات المقابلة
    np.save("embeddings/context_embeddings.npy", context_embeddings)
    with open("embeddings/unique_contexts.txt", "w", encoding="utf-8") as f:
        for context in unique_contexts:
            f.write(context + "\n")
    
    print("تم حفظ التمثيلات الرقمية والسياقات بنجاح!")

if __name__ == "__main__":
    main()