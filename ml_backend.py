# -*- coding: utf-8 -*-
"""
ML Backend for Media Perception Analysis
HF API Priority - Always use HF Space first, fallback only if needed
"""

import re
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import time

try:
    from gradio_client import Client as GradioClient
    GRADIO_CLIENT_AVAILABLE = True
except ImportError as e:
    GRADIO_CLIENT_AVAILABLE = False
    print(f"⚠️ gradio_client not available: {e}")

try:
    import torch
    from transformers import pipeline as hf_pipeline
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ PyTorch/Transformers not available: {e}")

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# HF API CONFIGURATION
# ============================================
# IMPORTANT: this must be the Space id (owner/name), NOT a hand-built REST URL.
# The Space is a Gradio app - it must expose the `api_predict` function via
# `gr.api(api_predict, api_name="predict")` inside the Blocks context on the
# Space side. Once that's added, gradio_client handles Gradio's queued call
# protocol correctly (a raw requests.post to /api/predict does NOT work
# reliably against modern Gradio and was the cause of the bad/neutral scores).
HF_SPACE_ID = "https://devxaman-sentiment-analyser.hf.space"
USE_HF_API = True

_hf_client = None


def get_hf_client():
    """Lazily create (and cache) the Gradio client connection to the HF Space."""
    global _hf_client
    if _hf_client is None:
        _hf_client = GradioClient(HF_SPACE_ID)
    return _hf_client


def analyze_with_hf(text):
    """Send text to HF Space for sentiment analysis via gradio_client"""
    if not text or len(text.strip()) < 5:
        return {
            'success': False,
            'error': 'Text too short',
            'sentiment': 'Neutral',
            'confidence': 0.5,
            'positive_score': 0.33,
            'negative_score': 0.33,
            'neutral_score': 0.34
        }

    if not GRADIO_CLIENT_AVAILABLE:
        return {
            'success': False,
            'error': 'gradio_client not installed',
            'sentiment': 'Neutral',
            'confidence': 0.5,
            'positive_score': 0.33,
            'negative_score': 0.33,
            'neutral_score': 0.34
        }

    try:
        client = get_hf_client()
        # This calls the `api_predict` function registered on the Space via
        # gr.api(api_predict, api_name="predict"). It already returns a dict:
        # {'sentiment', 'confidence', 'negative_score', 'neutral_score', 'positive_score'}
        result = client.predict(text=text[:512], api_name="/predict")

        sentiment = result.get('sentiment', 'Neutral')
        confidence = result.get('confidence', 0.5)
        positive_score = result.get('positive_score', 0.33)
        negative_score = result.get('negative_score', 0.33)
        neutral_score = result.get('neutral_score', 0.34)

        label = sentiment if sentiment in ('Positive', 'Negative', 'Neutral') else 'Neutral'
        score = {'Positive': positive_score, 'Negative': negative_score,
                  'Neutral': neutral_score}.get(label, confidence)

        return {
            'success': True,
            'sentiment': label,
            'confidence': float(score),
            'positive_score': float(positive_score),
            'negative_score': float(negative_score),
            'neutral_score': float(neutral_score),
            'source': 'hf_space',
            'raw_result': result
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'sentiment': 'Neutral',
            'confidence': 0.5,
            'positive_score': 0.33,
            'negative_score': 0.33,
            'neutral_score': 0.34
        }


class NewsAnalyzer:
    def __init__(self):
        logger.info("🚀 Initializing News Analyzer...")
        logger.info(f"🔗 HF Space: {HF_SPACE_ID}")
        logger.info(f"🤖 HF API Enabled: {USE_HF_API}")
        
        self.models = {}
        self._models_loaded = False
        
        if ML_AVAILABLE:
            self.device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if self.device == 0 else "CPU"
            logger.info(f"  -> Local ML device: {device_name} (loaded lazily on first local-fallback use)")
        else:
            logger.info("  -> Local ML not available")
        
        logger.info("✅ News Analyzer ready")
    
    def _load_models(self):
        if not ML_AVAILABLE or self._models_loaded:
            return
        self._models_loaded = True
        
        try:
            logger.info("  -> Loading distilbert-base-uncased-finetuned-sst-2-english...")
            self.models['general'] = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
                framework='pt',
                truncation=True,
                max_length=512
            )
            logger.info("     ✅ general model loaded")
        except Exception as e:
            logger.warning(f"     ⚠️ general model failed: {str(e)[:60]}")
            self.models['general'] = None
        
        if self.models.get('general') is not None:
            for domain in ['finance', 'defence', 'technology', 'politics']:
                self.models[domain] = self.models['general']
    
    def detect_domain(self, text):
        if not text:
            return 'general'
        
        text_lower = text.lower()
        domains = {
            'finance': ['revenue', 'profit', 'stock', 'market', 'investment', 'financial', 
                       'bank', 'fund', 'trading', 'shares', 'dividend', 'fiscal', 'economy',
                       'budget', 'tax', 'inflation', 'gdp'],
            'politics': ['government', 'minister', 'election', 'parliament', 'policy', 
                        'cabinet', 'political', 'vote', 'opposition', 'party', 'democracy',
                        'bjp', 'congress', 'modi', 'amit shah'],
            'defence': ['defence', 'defense', 'military', 'army', 'navy', 'weapon', 
                       'missile', 'security', 'combat', 'forces', 'warfare', 'strategic',
                       'drdo', 'hal', 'lca', 'tejas', 'rafale'],
            'technology': ['ai', 'software', 'tech', 'digital', 'innovation', 'algorithm',
                          'data', 'computing', 'internet', 'cyber', 'startup', 'platform']
        }
        
        scores = {domain: sum(1 for kw in keywords if kw in text_lower) 
                  for domain, keywords in domains.items()}
        
        max_score = max(scores.values())
        return max(scores, key=scores.get) if max_score > 0 else 'general'
    
    def analyze_single_local(self, text):
        """Fallback local ML analysis"""
        if not text or len(text.strip()) < 10:
            return {
                'sentiment_label': 'Neutral',
                'sentiment_score': 0.5,
                'detected_domain': 'general',
                'model_used': 'none'
            }
        
        domain = self.detect_domain(text)
        
        if ML_AVAILABLE and not self._models_loaded:
            self._load_models()
        
        if ML_AVAILABLE and self.models.get('general') is not None:
            try:
                text_to_analyze = text[:512]
                result = self.models['general'](text_to_analyze)[0]
                label = result['label']
                score = float(result['score'])
                
                label_lower = label.lower()
                if label_lower in ['positive', 'pos', '1']:
                    sentiment = 'Positive'
                elif label_lower in ['negative', 'neg', '0']:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                return {
                    'sentiment_label': sentiment,
                    'sentiment_score': round(score, 4),
                    'detected_domain': domain,
                    'model_used': 'local_fallback'
                }
            except Exception as e:
                logger.error(f"Local analysis failed: {str(e)[:80]}")
        
        return {
            'sentiment_label': 'Neutral',
            'sentiment_score': 0.5,
            'detected_domain': domain,
            'model_used': 'neutral_fallback'
        }
    
    def analyze_single_with_hf(self, text, allow_fallback=True):
        """Analyze using HF API. If allow_fallback is False (user forced HF-only
        mode), a failure is reported as-is instead of silently switching to the
        weaker local model."""
        result = analyze_with_hf(text)
        
        if result['success']:
            return {
                'sentiment_label': result['sentiment'],
                'sentiment_score': result['confidence'],
                'detected_domain': self.detect_domain(text),
                'model_used': 'hf_space',
                'positive_score': result['positive_score'],
                'negative_score': result['negative_score'],
                'neutral_score': result['neutral_score']
            }
        elif allow_fallback:
            logger.warning(f"HF API failed: {result.get('error')}, using local fallback")
            return self.analyze_single_local(text)
        else:
            logger.error(f"HF API failed (HF-only mode, no fallback): {result.get('error')}")
            return {
                'sentiment_label': 'Neutral',
                'sentiment_score': 0.5,
                'detected_domain': self.detect_domain(text),
                'model_used': 'hf_failed',
                'error': result.get('error', 'HF Space unreachable')
            }
    
    def analyze_batch(self, texts, mode='auto'):
        """Analyze batch of texts.
        mode: 'auto' (HF with local fallback, default), 'hf' (HF only, no
        silent fallback), or 'local' (skip HF entirely)."""
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if mode == 'local':
                result = self.analyze_single_local(text)
            elif mode == 'hf':
                result = self.analyze_single_with_hf(text, allow_fallback=False)
            else:  # auto
                result = self.analyze_single_with_hf(text, allow_fallback=True) if USE_HF_API else self.analyze_single_local(text)
            results.append(result)
            
            # Rate limiting for HF API
            if mode in ('auto', 'hf') and i % 5 == 0 and i > 0:
                time.sleep(0.5)
            
            if (i + 1) % 5 == 0 or (i + 1) == total:
                logger.info(f"  Processed {i + 1}/{total}")
        
        return results


def find_column(df, exact_names, contains_any=None, exclude_contains=None):
    """Robust column finder"""
    normalized = {c: c.strip().lower() for c in df.columns}

    for target in exact_names:
        t = target.strip().lower()
        for col, norm in normalized.items():
            if norm == t:
                return col

    if contains_any:
        exclude_contains = exclude_contains or ['id', 'code', 'url', 'link', 'attachment']
        for col, norm in normalized.items():
            if any(bad in norm for bad in exclude_contains):
                continue
            if any(needle in norm for needle in contains_any):
                return col

    return None


def process_excel_file(df, mode='auto'):
    """Process DataFrame - mode: 'auto' (HF w/ local fallback), 'hf' (HF only), 'local' (local only)"""
    analyzer = get_analyzer()
    
    logger.info("="*60)
    logger.info("📊 Processing Dataset")
    logger.info("="*60)
    logger.info(f"📋 Columns found: {list(df.columns)}")
    logger.info(f"📊 Total rows: {len(df)}")
    logger.info(f"🔗 Mode: {mode}")
    
    # Find text column
    text_col = find_column(
        df,
        exact_names=['Text(content)', 'content', 'text', 'body', 'article_text',
                     'description', 'summary', 'Text'],
        contains_any=['content', 'text', 'body', 'article']
    )
    headline_col = find_column(
        df,
        exact_names=['HeadLine', 'headline', 'title', 'Headline'],
        contains_any=['headline', 'title']
    )

    if text_col is None:
        raise ValueError(
            f"Could not find a content/text column. Columns: {list(df.columns)}"
        )

    if headline_col is None:
        headline_col = text_col

    logger.info(f"📰 Using '{headline_col}' as headline column")
    logger.info(f"📄 Using '{text_col}' as content column")

    # Prepare texts for analysis
    texts = []
    for idx, row in df.iterrows():
        headline = str(row[headline_col]) if headline_col in row and pd.notna(row[headline_col]) else ''
        content = str(row[text_col]) if text_col in row and pd.notna(row[text_col]) else ''
        combined = f"{headline} {content}".strip()
        texts.append(combined)

    # Run analysis using selected mode
    ml_results = analyzer.analyze_batch(texts, mode=mode)

    # Merge with original data
    results = []
    for idx, row in df.iterrows():
        item = row.to_dict()
        if idx < len(ml_results):
            item.update(ml_results[idx])

        item['headline'] = str(row[headline_col]) if pd.notna(row[headline_col]) else f"Article {idx+1}"
        item['content'] = str(row[text_col]) if pd.notna(row[text_col]) else ''

        # Ensure date
        if 'publication_date' not in item or not item['publication_date']:
            for col in ['DATE AND MONTH', 'date', 'Date', 'publication_date']:
                if col in row and pd.notna(row[col]):
                    item['publication_date'] = str(row[col])
                    break
            if 'publication_date' not in item:
                item['publication_date'] = datetime.now().strftime('%Y-%m-%d')

        # Ensure source
        if 'source' not in item or not item['source']:
            for col in ['NewsPaper/Magazine', 'source', 'Source']:
                if col in row and pd.notna(row[col]):
                    item['source'] = str(row[col])
                    break
            if 'source' not in item:
                item['source'] = 'Unknown'

        results.append(item)
    
    logger.info(f"✅ Analysis complete. Processed {len(results)} articles.")
    
    # Log model usage summary
    model_counts = {}
    for r in results:
        model = r.get('model_used', 'unknown')
        model_counts[model] = model_counts.get(model, 0) + 1
    logger.info(f"📊 Model usage: {model_counts}")
    
    return results


def process_news_data(df, mode='auto'):
    return process_excel_file(df, mode=mode)


def process_sql_query(query, db_path='news_analysis.db', mode='auto'):
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return []
        return process_excel_file(df, mode=mode)
    except Exception as e:
        conn.close()
        raise e


_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = NewsAnalyzer()
    return _analyzer
