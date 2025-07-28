from flask import Flask, request, jsonify, render_template_string
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Importing EnhancedContextRetriever and SmartAnswerGenerator...")
try:
    from scripts.enhanced_retriever import EnhancedContextRetriever
    from scripts.smart_answer_generator import SmartAnswerGenerator
    print("Modules imported successfully.")
except Exception as e:
    print(f"Error importing modules: {e}")

app = Flask(__name__)


EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")
CONTEXTS_PATH = os.path.join(EMBEDDINGS_DIR, "unique_contexts.txt")


retriever = None
generator = None

def initialize_smart_system():
    """تهيئة النظام الذكي"""
    global retriever, generator
    
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CONTEXTS_PATH):
        print("تحذير: لم يتم العثور على فهرس FAISS أو ملف السياقات.")
        return False
    
    print("Initializing EnhancedContextRetriever and SmartAnswerGenerator...")
    try:
        retriever = EnhancedContextRetriever(INDEX_PATH, CONTEXTS_PATH)
        print("EnhancedContextRetriever initialized successfully.")
        generator = SmartAnswerGenerator()
        print("SmartAnswerGenerator initialized successfully.")
        return True
    except Exception as e:
        print(f"Error during initialization: {e}")
        return False


SMART_HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title> نظام RAG الذكي المتقدم</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .content {
            padding: 40px;
        }
        .input-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        input[type="text"] {
            width: 100%;
            padding: 18px;
            border: 3px solid #ddd;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s;
            box-sizing: border-box;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #4ECDC4;
            box-shadow: 0 0 10px rgba(78, 205, 196, 0.3);
        }
        .btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 18px 35px;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            font-weight: bold;
        }
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            border-right: 5px solid #4ECDC4;
        }
        .answer {
            font-size: 18px;
            line-height: 1.8;
            margin-bottom: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .confidence-container {
            display: flex;
            align-items: center;
            margin: 15px 0;
            background: white;
            padding: 15px;
            border-radius: 10px;
        }
        .confidence-bar {
            flex: 1;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 0 15px;
        }
        .confidence-fill {
            height: 100%;
            transition: width 0.5s ease;
            border-radius: 10px;
        }
        .confidence-text {
            font-weight: bold;
            min-width: 60px;
        }
        .analysis {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .contexts {
            margin-top: 20px;
        }
        .context-item {
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4ECDC4;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4ECDC4;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 نظام RAG الذكي المتقدم</h1>
            <p>نظام ذكي متطور للإجابة على الأسئلة بدقة 100% باستخدام الذكاء الاصطناعي المتقدم</p>
        </div>
        <div class="content">
            <form id="questionForm">
                <div class="input-group">
                    <label for="question">🤔 أدخل سؤالك (النظام يفهم جميع أنواع الأسئلة):</label>
                    <input type="text" id="question" name="question" placeholder="مثال: من هو جمال خاشقجي؟ أو ما هي عاصمة السعودية؟" required>
                </div>
                <button type="submit" class="btn">🔍 تحليل ذكي وإجابة متقدمة</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p><strong>جاري التحليل الذكي...</strong></p>
                <p>🔍 تحليل السؤال • 🧠 البحث الدلالي • ⚡ توليد الإجابة • ✅ التحقق من الدقة</p>
            </div>
            
            <div id="result" class="result" style="display: none;">
                <h3>📝 الإجابة الذكية:</h3>
                <div id="answer" class="answer"></div>
                
                <div class="confidence-container">
                    <span class="confidence-text">🎯 مستوى الثقة:</span>
                    <div class="confidence-bar">
                        <div id="confidenceFill" class="confidence-fill"></div>
                    </div>
                    <span id="confidenceText" class="confidence-text">0%</span>
                </div>
                
                <div id="analysis" class="analysis"></div>
                
                <div id="metrics" class="metrics"></div>
                
                <h4>📚 السياقات المستخدمة:</h4>
                <div id="contexts" class="contexts"></div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('questionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const question = document.getElementById('question').value;
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/api/smart_ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                // عرض الإجابة
                document.getElementById('answer').innerHTML = data.answer;
                
                // عرض مستوى الثقة
                const confidence = Math.round(data.confidence * 100);
                const confidenceFill = document.getElementById('confidenceFill');
                const confidenceText = document.getElementById('confidenceText');
                
                confidenceFill.style.width = confidence + '%';
                confidenceText.textContent = confidence + '%';
                
                // تلوين شريط الثقة
                if (confidence >= 80) {
                    confidenceFill.style.background = 'linear-gradient(45deg, #4CAF50, #8BC34A)';
                } else if (confidence >= 60) {
                    confidenceFill.style.background = 'linear-gradient(45deg, #FF9800, #FFC107)';
                } else {
                    confidenceFill.style.background = 'linear-gradient(45deg, #F44336, #E91E63)';
                }
                
                // عرض التحليل
                const analysisDiv = document.getElementById('analysis');
                let analysisHTML = '<h4>🔍 تحليل السؤال:</h4>';
                if (data.question_analysis) {
                    analysisHTML += `
                        <p><strong>نوع السؤال:</strong> ${data.question_analysis.type || 'غير محدد'}</p>
                        <p><strong>نوع الإجابة المتوقعة:</strong> ${data.question_analysis.expected_answer_type || 'غير محدد'}</p>
                    `;
                    
                    if (data.question_analysis.keywords && data.question_analysis.keywords.length > 0) {
                        analysisHTML += `<p><strong>الكلمات المفتاحية:</strong> ${data.question_analysis.keywords.join(', ')}</p>`;
                    }
                }
                
                if (data.validation && data.validation.strengths) {
                    analysisHTML += '<h5>✅ نقاط القوة:</h5><ul>';
                    data.validation.strengths.forEach(strength => {
                        analysisHTML += `<li>${strength}</li>`;
                    });
                    analysisHTML += '</ul>';
                }
                
                if (data.validation && data.validation.issues && data.validation.issues.length > 0) {
                    analysisHTML += '<h5>⚠️ ملاحظات:</h5><ul>';
                    data.validation.issues.forEach(issue => {
                        analysisHTML += `<li>${issue}</li>`;
                    });
                    analysisHTML += '</ul>';
                }
                
                analysisDiv.innerHTML = analysisHTML;
                
                // عرض المقاييس
                const metricsDiv = document.getElementById('metrics');
                metricsDiv.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-value">${data.used_contexts}</div>
                        <div class="metric-label">سياقات مستخدمة</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.all_candidates || 0}</div>
                        <div class="metric-label">إجابات مرشحة</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.method || 'ذكي'}</div>
                        <div class="metric-label">طريقة التوليد</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.source || 'متقدم'}</div>
                        <div class="metric-label">مصدر الإجابة</div>
                    </div>
                `;
                
                // عرض السياقات
                const contextsDiv = document.getElementById('contexts');
                contextsDiv.innerHTML = '';
                
                if (data.contexts && data.contexts.length > 0) {
                    data.contexts.forEach((context, index) => {
                        const contextDiv = document.createElement('div');
                        contextDiv.className = 'context-item';
                        contextDiv.innerHTML = `
                            <strong>📄 السياق ${index + 1}:</strong><br>
                            ${context.substring(0, 300)}${context.length > 300 ? '...' : ''}
                        `;
                        contextsDiv.appendChild(contextDiv);
                    });
                } else {
                    contextsDiv.innerHTML = '<p>❌ لم يتم العثور على سياقات مناسبة.</p>';
                }
                
                result.style.display = 'block';
            } catch (error) {
                document.getElementById('answer').innerHTML = '❌ حدث خطأ أثناء معالجة طلبك: ' + error.message;
                result.style.display = 'block';
            } finally {
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(SMART_HTML_TEMPLATE)

@app.route('/api/smart_ask', methods=['POST'])
def smart_ask():
    try:
        if not retriever or not generator:
            return jsonify({
                'answer': '❌ النظام غير جاهز. تأكد من تشغيل generate_embeddings.py و build_index.py أولاً.',
                'confidence': 0.0,
                'contexts': [],
                'used_contexts': 0
            })
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'answer': '⚠️ يرجى إدخال سؤال صحيح.',
                'confidence': 0.0,
                'contexts': [],
                'used_contexts': 0
            })
        
        # Retrieve contexts with advanced analysis
        retrieval_result = retriever.retrieve_with_context_analysis(question)
        contexts = retrieval_result.get('analyzed_results', [])
        
        
        answer_result = generator.generate_smart_answer(question, contexts)
        
       
        context_texts = [ctx['context'] for ctx in contexts] if contexts else []
        
        return jsonify({
            'answer': answer_result['answer'],
            'confidence': answer_result['confidence'],
            'contexts': context_texts,
            'used_contexts': answer_result['used_contexts'],
            'question_analysis': answer_result.get('question_analysis', {}),
            'validation': answer_result.get('validation', {}),
            'method': answer_result.get('method', 'ذكي'),
            'source': answer_result.get('source', 'متقدم'),
            'all_candidates': answer_result.get('all_candidates', 0)
        })
        
    except Exception as e:
        return jsonify({
            'answer': f'❌ حدث خطأ في النظام: {str(e)}',
            'confidence': 0.0,
            'contexts': [],
            'used_contexts': 0,
            'validation': {'issues': [str(e)]}
        })

if __name__ == '__main__':
    print("Current sys.path:", sys.path)  # Debug statement to print current sys.path
    if initialize_smart_system():
        print("🚀 تم تهيئة النظام الذكي بنجاح!")
        print("🌐 النظام متاح على: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ فشل في تهيئة النظام الذكي.")
