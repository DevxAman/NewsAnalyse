# 📊 NewsAnalyze - PDF Sentiment Analysis Platform

Professional-grade web application for analyzing sentiment in PDF news articles using domain-aware ML models.

## 🚀 Features

- **Multi-Domain Sentiment Analysis**: Automatic domain detection (Finance, Politics, Defence, Technology)
- **Advanced ML Models**:
  - 📈 **FinBERT**: Finance domain specialist
  - 🗣️ **RoBERTa**: Politics & general news
  - 💪 **Siebert**: Defence & technology
  - 🎯 **Outcome Detection**: Smart positive/negative outcome detection
- **PDF Processing**: Robust text extraction with multiple fallback methods
- **Beautiful UI**: Modern, responsive web interface
- **Export Results**: Download analysis as CSV

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) Tesseract OCR for scanned PDFs

## 🛠️ Installation

1. **Navigate to project directory**:
```bash
cd "C:\Users\AMANDEEP SINGH\.gemini\antigravity\scratch\NewsAnalyze"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: First run will download ML models (~2GB). This may take several minutes.

## 🎯 Usage

### Start the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### Using the Application

1. Open your browser and navigate to `http://localhost:5000`
2. Click "Choose Files" and select PDF news articles
3. Click "🚀 Start Analysis with ML Models"
4. Wait for processing (may take time for first run as models load)
5. View results in the interactive table
6. Download results as CSV if needed

## 📁 Project Structure

```
NewsAnalyze/
├── app.py                  # Flask application server
├── ml_backend.py           # ML processing pipeline
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Frontend UI
└── uploads/               # Temporary file storage
```

## 🧠 How It Works

1. **PDF Upload**: User uploads PDF files via web interface
2. **Text Extraction**: Multi-method extraction (pdfplumber → PyPDF2 → OCR)
3. **Domain Detection**: Keyword-based domain classification
4. **Sentiment Analysis**: Domain-specific model selection
5. **Results Display**: Interactive table with statistics

## 🔧 API Endpoints

### `GET /`
Serves the main web interface

### `POST /api/analyze`
Analyzes uploaded PDF files

**Request**: `multipart/form-data` with PDF files
**Response**: JSON with analysis results

```json
{
  "results": [
    {
      "file_name": "article.pdf",
      "headline": "...",
      "publication_date": "2024-01-04",
      "article_content": "...",
      "detected_domain": "finance",
      "sentiment_label": "Positive",
      "sentiment_score": 0.8542,
      "sentiment_model_used": "finbert"
    }
  ]
}
```

## ⚠️ Important Notes

- **First Run**: Models download automatically (~2GB). Be patient!
- **GPU Support**: Automatically uses GPU if available (CUDA)
- **Memory**: Requires ~4GB RAM for model loading
- **OCR**: Install Tesseract for scanned PDF support (optional)

## 🐛 Troubleshooting

### Models not loading?
- Check internet connection (models download from HuggingFace)
- Ensure sufficient disk space (~3GB)

### OCR not working?
- Install Tesseract: `https://github.com/tesseract-ocr/tesseract`
- Set path in `ml_backend.py` if needed

### Import errors?
```bash
pip install --upgrade -r requirements.txt
```

## 📊 Supported Domains

- **Finance**: Revenue, stocks, markets, banking
- **Politics**: Government, elections, policies
- **Defence**: Military, weapons, security
- **Technology**: AI, software, startups
- **General**: Fallback for other topics

## 🎨 Tech Stack

- **Backend**: Flask, Python
- **ML**: Transformers, PyTorch, FinBERT, RoBERTa, Siebert
- **PDF**: pdfplumber, PyPDF2, pytesseract
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## 📝 License

Educational/Research Project

---

**Built with ❤️ for professional sentiment analysis**
