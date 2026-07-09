"""
HuggingFace Space App - News Sentiment Analyzer
ZeroGPU Compatible Version - torch 2.8.0
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

print("🚀 Loading Sentiment Models...")

# Device configuration - ZeroGPU compatible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Using device: {device}")

# ============================================
# 1. LOAD ROBERTA MODEL
# ============================================
print("📥 Loading RoBERTa model...")
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name).to(device)
roberta_model.eval()
print("✅ RoBERTa model loaded!")

# ============================================
# 2. LOAD SIEBERT MODEL
# ============================================
print("📥 Loading Siebert model...")
siebert_model_name = "siebert/sentiment-roberta-large-english"
siebert_tokenizer = AutoTokenizer.from_pretrained(siebert_model_name)
siebert_model = AutoModelForSequenceClassification.from_pretrained(siebert_model_name).to(device)
siebert_model.eval()
print("✅ Siebert model loaded!")

print("🎯 All models ready!")


# ============================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================

def analyze_roberta(text):
    """Analyze sentiment using RoBERTa model"""
    try:
        if not text or len(text.strip()) < 5:
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
        
        encoded = roberta_tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            output = roberta_model(**encoded)
        
        scores = softmax(output.logits.cpu().numpy()[0])
        return {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2])
        }
    except Exception as e:
        print(f"❌ RoBERTa error: {e}")
        return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}


def analyze_siebert(text):
    """Analyze sentiment using Siebert model"""
    try:
        if not text or len(text.strip()) < 5:
            return {'negative': 0.5, 'positive': 0.5}
        
        encoded = siebert_tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            output = siebert_model(**encoded)
        
        scores = softmax(output.logits.cpu().numpy()[0])
        return {
            'negative': float(scores[0]),
            'positive': float(scores[1])
        }
    except Exception as e:
        print(f"❌ Siebert error: {e}")
        return {'negative': 0.5, 'positive': 0.5}


def get_ensemble_sentiment(text):
    """Combine both models for accurate results"""
    
    roberta_result = analyze_roberta(text)
    siebert_result = analyze_siebert(text)
    
    siebert_neg = siebert_result['negative']
    siebert_pos = siebert_result['positive']
    diff = abs(siebert_pos - siebert_neg)
    
    if diff < 0.2:
        siebert_neutral = 1 - diff
        siebert_neg = siebert_neg * (1 - siebert_neutral/2)
        siebert_pos = siebert_pos * (1 - siebert_neutral/2)
    else:
        siebert_neutral = 0.1
        siebert_neg = siebert_neg * 0.95
        siebert_pos = siebert_pos * 0.95
    
    ensemble_neg = (roberta_result['negative'] + siebert_neg) / 2
    ensemble_neu = (roberta_result['neutral'] + siebert_neutral) / 2
    ensemble_pos = (roberta_result['positive'] + siebert_pos) / 2
    
    total = ensemble_neg + ensemble_neu + ensemble_pos
    if total > 0:
        ensemble_neg /= total
        ensemble_neu /= total
        ensemble_pos /= total
    
    max_score = max(ensemble_neg, ensemble_neu, ensemble_pos)
    if max_score == ensemble_neg:
        label = 'Negative'
    elif max_score == ensemble_pos:
        label = 'Positive'
    else:
        label = 'Neutral'
    
    return {
        'sentiment': label,
        'confidence': round(max_score, 4),
        'negative_score': round(ensemble_neg, 4),
        'neutral_score': round(ensemble_neu, 4),
        'positive_score': round(ensemble_pos, 4)
    }


def predict_single(text):
    """Predict sentiment for single text"""
    if not text or len(text.strip()) < 5:
        return "⚠️ Please enter some text to analyze."
    
    result = get_ensemble_sentiment(text)
    
    sentiment_emoji = {'Positive': '✅', 'Negative': '❌', 'Neutral': '⚪'}
    emoji = sentiment_emoji.get(result['sentiment'], '⚪')
    confidence_pct = result['confidence'] * 100
    
    if confidence_pct > 80:
        confidence_color = "🟢"
    elif confidence_pct > 60:
        confidence_color = "🟡"
    else:
        confidence_color = "🔴"
    
    return f"""
# 📊 Sentiment Analysis Results

---

### {emoji} **Final Sentiment: {result['sentiment']}**
**Confidence:** {confidence_color} {confidence_pct:.1f}%

---

### 📈 Score Breakdown

| Aspect | Score |
|--------|-------|
| ✅ Positive | {result['positive_score']*100:.1f}% |
| ❌ Negative | {result['negative_score']*100:.1f}% |
| ⚪ Neutral | {result['neutral_score']*100:.1f}% |

---

### 📝 Analyzed Text
> {text[:200]}{'...' if len(text) > 200 else ''}
"""


def api_predict(text):
    """API endpoint for external calls (JSON response)"""
    if not text or len(text.strip()) < 5:
        return {
            'error': 'Text too short or empty',
            'sentiment': 'Neutral',
            'confidence': 0.0
        }
    
    return get_ensemble_sentiment(text)


# ============================================
# GRADIO INTERFACE
# ============================================

# Create the interface
demo = gr.Interface(
    fn=predict_single,
    inputs=gr.Textbox(
        label="📝 Enter News Text",
        placeholder="Paste your news article here...",
        lines=8,
        max_lines=20
    ),
    outputs=gr.Markdown(label="📈 Analysis Results"),
    title="📰 News Sentiment Analyzer",
    description="""
    ### 🎯 Ensemble Model: RoBERTa + Siebert
    
    This analyzer uses two powerful models combined for accurate sentiment analysis:
    - **RoBERTa**: Twitter-based sentiment model (negative, neutral, positive)
    - **Siebert**: Large-scale sentiment model (negative, positive)
    
    The ensemble approach provides more robust results by combining both models.
    """,
    examples=[
        ["Apple reported record profits with revenue growth of 15% this quarter. The tech giant exceeded market expectations."],
        ["The company faces severe criticism over data breach affecting millions of users. Stocks fell sharply."],
        ["Government announced new policies to boost economic growth and create jobs. Markets reacted positively."],
        ["Defense ministry successfully tested new hypersonic missile system."],
        ["The bank reported $2 billion in losses due to failed investments. CEO resigned."]
    ],
    theme="soft",
    cache_examples=True
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 News Sentiment Analyzer")
    print("="*60)
    print("📱 Device:", device)
    print("🤖 Models: RoBERTa + Siebert")
    print("="*60 + "\n")
    
    demo.launch(share=True)