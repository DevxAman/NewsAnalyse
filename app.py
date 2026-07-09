import os
import secrets
import json
import requests
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import sqlite3
from datetime import datetime
import io
import sys

# Try to import ml_backend for local fallback
try:
    from ml_backend import process_news_data, process_excel_file, process_sql_query
    ML_BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: ml_backend import failed: {e}")
    ML_BACKEND_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'news_analysis.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

# ============================================
# HF SPACE CONFIGURATION
# ============================================
# Kept here only for display/status purposes - the actual call to the Space
# goes through gradio_client inside ml_backend.py (see get_hf_client()).
HF_SPACE_ID = "DevxAman/Sentiment_Analyser"
USE_HF_API = True  # ALWAYS use HF API

# ============================================
# DATABASE
# ============================================
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT,
            content TEXT,
            publication_date TEXT,
            source TEXT,
            category TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            detected_domain TEXT,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def clear_database():
    """Clear all data from the database"""
    conn = sqlite3.connect(app.config['DATABASE'])
    try:
        c = conn.cursor()
        c.execute('DELETE FROM news_articles')
        conn.commit()
        print("🗑️ Database cleared")
        return True
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False
    finally:
        conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================
# FLASK ROUTES
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    """Check HF Space status"""
    status = {
        'hf_api': {'available': False, 'space_id': HF_SPACE_ID},
        'local_ml': {'available': ML_BACKEND_AVAILABLE},
        'use_hf': USE_HF_API
    }

    try:
        from ml_backend import analyze_with_hf
        start = datetime.now()
        result = analyze_with_hf("Test connection for status check.")
        status['hf_api']['available'] = bool(result.get('success'))
        status['hf_api']['response_time'] = round((datetime.now() - start).total_seconds(), 2)
        if not result.get('success'):
            status['hf_api']['error'] = result.get('error')
    except Exception as e:
        status['hf_api']['error'] = str(e)

    return jsonify(status)

@app.route('/api/clear', methods=['POST'])
def clear_database_route():
    """Clear all data from database"""
    try:
        success = clear_database()
        if success:
            return jsonify({'success': True, 'message': 'Database cleared successfully'})
        else:
            return jsonify({'error': 'Failed to clear database'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/excel', methods=['POST'])
def analyze_excel():
    """Analyze news from Excel/CSV file using HF API"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload XLSX, XLS, or CSV'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        mode = request.form.get('mode', 'auto')
        if mode not in ('auto', 'hf', 'local'):
            mode = 'auto'

        # Process with ml_backend (which now uses HF API)
        if ML_BACKEND_AVAILABLE:
            try:
                results = process_excel_file(df, mode=mode)
            except ValueError as e:
                return jsonify({'error': f'Could not process this file: {e}'}), 400
            except Exception as e:
                print(f"Processing error: {e}")
                return jsonify({'error': f'Analysis failed: {e}'}), 500
        else:
            return jsonify({'error': 'ML backend not available'}), 503
        
        # Clear old data and store new results
        clear_database()
        store_results_in_db(results)
        
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'results': results,
            'total': len(results),
            'summary': generate_summary(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/sql', methods=['POST'])
def analyze_sql():
    """Analyze news from SQL query using HF API"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'SQL query required'}), 400
    
    query = data['query']
    mode = data.get('mode', 'auto')
    if mode not in ('auto', 'hf', 'local'):
        mode = 'auto'
    
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return jsonify({'error': 'No results found'}), 400
        
        if ML_BACKEND_AVAILABLE:
            try:
                results = process_excel_file(df, mode=mode)
            except Exception as e:
                print(f"Processing error: {e}")
                return jsonify({'error': f'Analysis failed: {e}'}), 500
        else:
            return jsonify({'error': 'ML backend not available'}), 503
        
        return jsonify({
            'results': results,
            'total': len(results),
            'summary': generate_summary(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    conn = sqlite3.connect(app.config['DATABASE'])
    try:
        df = pd.read_sql_query('''
            SELECT headline, publication_date, source, sentiment_label, 
                   sentiment_score, detected_domain, model_used, created_at
            FROM news_articles
            ORDER BY created_at DESC
            LIMIT 100
        ''', conn)
        conn.close()
        return jsonify(df.to_dict('records'))
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get aggregated dashboard data"""
    conn = sqlite3.connect(app.config['DATABASE'])
    try:
        sentiment_counts = pd.read_sql_query('''
            SELECT sentiment_label, COUNT(*) as count
            FROM news_articles
            WHERE sentiment_label IS NOT NULL AND sentiment_label != ''
            GROUP BY sentiment_label
        ''', conn)
        
        domain_counts = pd.read_sql_query('''
            SELECT detected_domain, COUNT(*) as count
            FROM news_articles
            WHERE detected_domain IS NOT NULL AND detected_domain != ''
            GROUP BY detected_domain
        ''', conn)
        
        model_counts = pd.read_sql_query('''
            SELECT model_used, COUNT(*) as count
            FROM news_articles
            WHERE model_used IS NOT NULL
            GROUP BY model_used
        ''', conn)
        
        total = pd.read_sql_query('SELECT COUNT(*) as count FROM news_articles', conn).iloc[0, 0]
        
        conn.close()
        
        return jsonify({
            'sentiment_distribution': sentiment_counts.to_dict('records'),
            'domain_distribution': domain_counts.to_dict('records'),
            'model_distribution': model_counts.to_dict('records'),
            'total_articles': total
        })
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

def store_results_in_db(results):
    """Store analysis results in database"""
    conn = sqlite3.connect(app.config['DATABASE'])
    try:
        for item in results:
            conn.execute('''
                INSERT INTO news_articles 
                (headline, content, publication_date, source, category, 
                 sentiment_label, sentiment_score, detected_domain, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(item.get('headline', ''))[:200],
                str(item.get('content', ''))[:5000],
                str(item.get('publication_date', '')),
                str(item.get('source', '')),
                str(item.get('category', '')),
                str(item.get('sentiment_label', '')),
                float(item.get('sentiment_score', 0)) if item.get('sentiment_score') else 0,
                str(item.get('detected_domain', '')),
                str(item.get('model_used', ''))
            ))
        conn.commit()
        print(f"✅ Stored {len(results)} articles in database")
    except Exception as e:
        print(f"Error storing results: {e}")
    finally:
        conn.close()

def generate_summary(results):
    """Generate summary statistics"""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    total = len(df)
    
    if total == 0:
        return {}
    
    positive = len(df[df['sentiment_label'] == 'Positive']) if 'sentiment_label' in df else 0
    negative = len(df[df['sentiment_label'] == 'Negative']) if 'sentiment_label' in df else 0
    neutral = len(df[df['sentiment_label'] == 'Neutral']) if 'sentiment_label' in df else 0
    
    domains = {}
    if 'detected_domain' in df:
        domains = df['detected_domain'].value_counts().to_dict()
    
    models = {}
    if 'model_used' in df:
        models = df['model_used'].value_counts().to_dict()
    
    avg_score = 0
    if 'sentiment_score' in df:
        avg_score = df['sentiment_score'].mean()
    
    return {
        'total': total,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'domains': domains,
        'models': models,
        'avg_score': float(avg_score) if avg_score else 0
    }

@app.route('/api/export', methods=['GET'])
def export_results():
    """Export all results as Excel"""
    conn = sqlite3.connect(app.config['DATABASE'])
    try:
        df = pd.read_sql_query('SELECT * FROM news_articles ORDER BY created_at DESC', conn)
        conn.close()
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='News Analysis')
        
        output.seek(0)
        return send_file(
            output, 
            download_name=f'news_analysis_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx', 
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("📊 NewsAnalyze - Media Perception Dashboard")
    print("=" * 60)
    print(f"🚀 Server running on 0.0.0.0:{port}")
    print(f"📁 Database: {app.config['DATABASE']}")
    print("=" * 60)
    print(f"🔗 HF Space: {HF_SPACE_ID}")
    print(f"🤖 HF API Enabled: {USE_HF_API}")
    print(f"🧠 Local ML Available: {ML_BACKEND_AVAILABLE}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
