<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Emotion Recognition from Audio & Text - Documentation</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      line-height: 1.7;
      color: #333;
      max-width: 900px;
      margin: 40px auto;
      padding: 20px;
      background-color: #f9f9fb;
    }
    h1, h2, h3 {
      color: #2c3e50;
    }
    h1 {
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
    }
    h2 {
      margin-top: 30px;
      color: #2980b9;
    }
    code {
      background-color: #f0f0f0;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      font-size: 0.95em;
    }
    pre {
      background-color: #f5f5f5;
      padding: 15px;
      border-radius: 6px;
      overflow-x: auto;
      border: 1px solid #ddd;
    }
    ul, ol {
      margin: 15px 0;
    }
    a {
      color: #3498db;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .footer {
      margin-top: 50px;
      text-align: center;
      font-size: 0.9em;
      color: #7f8c8d;
      border-top: 1px solid #eee;
      padding-top: 20px;
    }
    .highlight {
      background-color: #fffacd;
      padding: 2px 6px;
      border-left: 4px solid #ffd700;
    }
  </style>
</head>
<body>

  <h1>🎙️ Multimodal Emotion Recognition System (Audio + Text)</h1>

  <p>
    A robust deep learning system that detects emotions from spoken audio and corresponding Bengali text using <strong>Wav2Vec2</strong> and <strong>Bangla-BERT</strong> features, fused and classified via an ensemble of <strong>MLP</strong> and <strong>BiGRU</strong> models.
  </p>

  <h2>🚀 Overview</h2>
  <p>
    This application combines acoustic and linguistic cues to classify emotional states in Bengali speech. It uses state-of-the-art transformer-based models for feature extraction and deep neural networks for classification, achieving high accuracy in multimodal emotion recognition.
  </p>

  <h2>📁 Project Structure</h2>
  <pre>
emotional-recognition-app/
│
├── models/
│   ├── mlp_no_pca.pth           # Trained MLP model
│   ├── bigru_no_pca.pth         # Trained BiGRU model
│   └── emotion_map_no_pca.npy   # Emotion label mapping
│
├── templates/
│   ├── index.html               # Homepage
│   ├── predict.html             # Prediction interface
│   └── about.html               # About page
│
├── static/
│   └── style.css                # Optional styling
│
├── app.py                       # Main FastAPI backend
├── README.html                  # This documentation
└── requirements.txt             # Dependencies
  </pre>

  <h2>🔧 Key Technologies</h2>
  <ul>
    <strong>Backend:</strong>
    <li><a href="https://fastapi.tiangolo.com/" target="_blank">FastAPI</a> – High-performance web framework for APIs</li>
    <li><a href="https://pytorch.org/" target="_blank">PyTorch</a> – Deep learning framework</li>
  </ul>
  <ul>
    <strong>Models:</strong>
    <li><a href="https://huggingface.co/facebook/wav2vec2-base" target="_blank">Wav2Vec2</a> – For audio feature extraction</li>
    <li><a href="https://huggingface.co/sagorsarker/bangla-bert-base" target="_blank">Bangla-BERT</a> – For Bengali text encoding</li>
    <li>MLP (Multi-Layer Perceptron) and BiGRU (Bidirectional GRU) – Classification models</li>
  </ul>
  <ul>
    <strong>Visualization:</strong>
    <li><a href="https://matplotlib.org/" target="_blank">Matplotlib</a> & <a href="https://seaborn.pydata.org/" target="_blank">Seaborn</a> – Probability distribution plots</li>
  </ul>

  <h2>🎯 Supported Emotions</h2>
  <p>
    The system classifies speech into one of the following <span class="highlight">6 emotion classes</span>:
  </p>
  <ul>
    <li>Happy</li>
    <li>Sad</li>
    <li>Angry</li>
    <li>Fear</li>
    <li>Surprise</li>
    <li>Neutral</li>
  </ul>

  <h2>⚙️ How It Works</h2>
  <ol>
    <li><strong>Audio Input:</strong> User uploads a .wav or .mp3 audio file.</li>
    <li><strong>Text Input:</strong> User provides transcribed Bengali text.</li>
    <li><strong>Preprocessing:</strong>
      <ul>
        <li>Audio is resampled to 16kHz, trimmed, and padded to 7 seconds.</li>
        <li>Text is cleaned and normalized (Unicode Bengali only).</li>
      </ul>
    </li>
    <li><strong>Feature Extraction:</strong>
      <ul>
        <li>Acoustic features via <code>wav2vec2-base</code></li>
        <li>Linguistic features via <code>sagorsarker/bangla-bert-base</code></li>
      </ul>
    </li>
    <li><strong>Fusion:</strong> Audio and text features are concatenated (1536-dim vector).</li>
    <li><strong>Prediction:</strong> Ensemble of MLP and BiGRU models (weighted 0.6 : 0.4) produces final emotion and confidence.</li>
    <li><strong>Visualization:</strong> Probability bar chart is generated and returned.</li>
  </ol>

  <h2>🌐 API Endpoints</h2>
  <ul>
    <li><code>GET /</code> – Home page</li>
    <li><code>GET /about</code> – Project details</li>
    <li><code>GET /predict</code> – Load prediction interface</li>
    <li><code>POST /predict</code> – Accepts audio file and text, returns emotion results and plot</li>
  </ul>

  <h2>📤 Example Request (via curl)</h2>
  <pre>
curl -X POST \
  http://localhost:8000/predict \
  -F "audio=@sample.wav" \
  -F "text='আমি খুব খুশি আজকে'"
  </pre>

  <h2>📤 Example Response</h2>
  <pre>
{
  "results": {
    "MLP": { "emotion": "Happy", "confidence": 0.92, "probs": [...] },
    "BiGRU": { "emotion": "Happy", "confidence": 0.88, "probs": [...] },
    "Ensemble": { "emotion": "Happy", "confidence": 0.904, "probs": [...] }
  },
  "plot": "base64_encoded_png",
  "audio": "base64_encoded_audio",
  "emotion_classes": ["Happy", "Sad", ...]
}
  </pre>

  <h2>📦 Requirements</h2>
  <p>Install dependencies using:</p>
  <pre>
pip install torch torchaudio fastapi uvicorn transformers librosa numpy matplotlib seaborn jinja2 python-multipart
  </pre>

  <h2>🚀 Running the App</h2>
  <pre>
uvicorn app:app --reload --host 0.0.0.0 --port 8000
  </pre>
  <p>Visit <a href="http://localhost:8000">http://localhost:8000</a> to access the web interface.</p>

  <h2>🛠️ Customization Tips</h2>
  <ul>
    <li>Replace models in <code>models/</code> to use different trained weights.</li>
    <li>Modify <code>ensemble_weights</code> in <code>EnsemblePredictor</code> to adjust model contribution.</li>
    <li>Update <code>emotion_map.npy</code> if emotion labels change.</li>
    <li>Add more visualizations or export options (e.g., JSON download).</li>
  </ul>

  <h2>📄 License</h2>
  <p>This project is for research and educational use. Refer to model licenses on Hugging Face for usage rights of Wav2Vec2 and Bangla-BERT.</p>

  <div class="footer">
    Developed with ❤️ for multimodal emotion analysis in Bengali. 
    <br>For research, academic, or collaboration inquiries, contact kaif.khan@northsouth.edu.
  </div>

</body>
</html>
