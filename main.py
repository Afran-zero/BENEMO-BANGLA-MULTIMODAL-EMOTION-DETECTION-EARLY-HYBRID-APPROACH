import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load emotion map
LOCAL_OUTPUT_DIR = "models"
emotion_map = np.load(os.path.join(LOCAL_OUTPUT_DIR, "emotion_map_no_pca.npy"), allow_pickle=True).item()
emotion_classes = list(emotion_map.keys())
num_classes = len(emotion_classes)

# Load feature extractors
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
bert_tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
bert_model = AutoModel.from_pretrained("sagorsarker/bangla-bert-base").to(device).eval()

# Audio preprocessing
def preprocess_audio(audio_data, target_sr=16000, duration=7):
    try:
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=None)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        target_length = target_sr * duration
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Text preprocessing
def preprocess_text(text):
    if not text or text.strip() == '':
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\u0980-\u09FF\w\s।,?!]', '', text)
    return text

# Feature extraction
def extract_audio_features(audio, processor, model):
    try:
        audio_tensor = torch.tensor(audio, dtype=torch.float32).to(device)
        inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def extract_text_features(text, tokenizer, model):
    try:
        if not text or text.strip() == '':
            print("Empty text provided, using zero features")
            return np.zeros(768)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error extracting text features: {e}")
        return None

# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, output_dim=6, num_layers=3, dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
        for _ in range(num_layers - 2):
            next_dim = current_dim // 2
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# BiGRU model definition
class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.layer_norm(out)
        out = self.fc(out)
        return out

# Load models
def load_mlp_model():
    model_path = 'models/mlp_no_pca.pth'
    mlp_model = MLP(input_dim=1536, hidden_dim=1024, output_dim=num_classes, num_layers=3, dropout=0.5).to(device)
    state_dict = torch.load(model_path, map_location=device)
    mlp_model.load_state_dict(state_dict)
    mlp_model.eval()
    return mlp_model

def load_bigru_model():
    model_path = 'models/bigru_no_pca.pth'
    bigru_model = BiGRU(input_dim=1536, hidden_dim=512, output_dim=num_classes, num_layers=3, dropout=0.5).to(device)
    state_dict = torch.load(model_path, map_location=device)
    bigru_model.load_state_dict(state_dict)
    bigru_model.eval()
    return bigru_model

# Ensemble predictor
class EnsemblePredictor:
    def __init__(self, mlp_model, bigru_model, ensemble_weights=None):
        self.mlp_model = mlp_model
        self.bigru_model = bigru_model
        self.weights = ensemble_weights if ensemble_weights else [0.5, 0.5]

    def predict(self, feature, emotion_map, device):
        predictions = {}
        probabilities = {}
        if self.mlp_model:
            mlp_emotion, mlp_probs, mlp_conf = self._predict_single_model(
                self.mlp_model, feature, emotion_map, device, "MLP")
            predictions['MLP'] = {'emotion': mlp_emotion, 'confidence': float(mlp_conf), 'probs': mlp_probs.tolist()}
            probabilities['MLP'] = mlp_probs
        if self.bigru_model:
            bigru_emotion, bigru_probs, bigru_conf = self._predict_single_model(
                self.bigru_model, feature, emotion_map, device, "BiGRU")
            predictions['BiGRU'] = {'emotion': bigru_emotion, 'confidence': float(bigru_conf), 'probs': bigru_probs.tolist()}
            probabilities['BiGRU'] = bigru_probs
        if len(probabilities) > 1:
            ensemble_probs = self._ensemble_probabilities(probabilities)
            ensemble_class = np.argmax(ensemble_probs)
            reverse_emotion_map = {v: k for k, v in emotion_map.items()}
            ensemble_emotion = reverse_emotion_map.get(ensemble_class, "unknown")
            ensemble_conf = ensemble_probs[ensemble_class]
            predictions['Ensemble'] = {
                'emotion': ensemble_emotion,
                'confidence': float(ensemble_conf),
                'probs': ensemble_probs.tolist()
            }
        return predictions

    def _predict_single_model(self, model, feature, emotion_map, device, model_name):
        model.eval()
        try:
            if len(feature.shape) == 1:
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                feature = torch.tensor(feature, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(feature)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                reverse_emotion_map = {v: k for k, v in emotion_map.items()}
                emotion = reverse_emotion_map.get(predicted_class, "unknown")
                confidence = probabilities[predicted_class]
                return emotion, probabilities, confidence
        except Exception as e:
            print(f"Error in {model_name} prediction: {e}")
            return None, None, None

    def _ensemble_probabilities(self, probabilities):
        model_names = list(probabilities.keys())
        weighted_probs = np.zeros_like(probabilities[model_names[0]])
        for i, model_name in enumerate(model_names):
            weight = self.weights[i]
            weighted_probs += weight * probabilities[model_name]
        return weighted_probs

# Load models
mlp_model = load_mlp_model()
bigru_model = load_bigru_model()
ensemble_predictor = EnsemblePredictor(mlp_model, bigru_model, ensemble_weights=[0.6, 0.4])

# Generate plot
def generate_plot(results, emotion_classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_classes)))
    result = results['Ensemble']
    bars = ax.bar(emotion_classes, result['probs'], color=colors)
    for bar, prob in zip(bars, result['probs']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.set_title(f'Ensemble Prediction: {result["emotion"]} ({result["confidence"]:.3f})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Emotion', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_xticks(range(len(emotion_classes)))  # Set tick positions
    ax.set_xticklabels(emotion_classes, rotation=45)  # Set tick labels
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    return base64.b64encode(image_png).decode('utf-8')

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict(audio: UploadFile = File(...), text: str = Form(...)):
    try:
        audio_data = await audio.read()
        text_input = preprocess_text(text)
        audio_processed = preprocess_audio(audio_data)
        if audio_processed is None:
            raise HTTPException(status_code=400, detail="Failed to process audio")
        audio_features = extract_audio_features(audio_processed, wav2vec_processor, wav2vec_model)
        if audio_features is None:
            raise HTTPException(status_code=400, detail="Failed to extract audio features")
        text_features = extract_text_features(text_input, bert_tokenizer, bert_model)
        if text_features is None:
            raise HTTPException(status_code=400, detail="Failed to extract text features")
        combined_features = np.concatenate([audio_features, text_features])
        results = ensemble_predictor.predict(combined_features, emotion_map, device)
        if not results:
            raise HTTPException(status_code=500, detail="Prediction failed")
        plot_base64 = generate_plot(results, emotion_classes)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        return {
            "results": results,
            "plot": plot_base64,
            "audio": audio_base64,
            "emotion_classes": emotion_classes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))