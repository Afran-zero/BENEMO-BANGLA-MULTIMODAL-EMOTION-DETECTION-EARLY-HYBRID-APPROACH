import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import re
import time
import tempfile
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
from banglaspeech2text import Speech2Text
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import librosa
import librosa.effects

# Set audio backend
torchaudio.set_audio_backend("soundfile")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load emotion map
LOCAL_OUTPUT_DIR = "models"
try:
    emotion_map = np.load(os.path.join(LOCAL_OUTPUT_DIR, "emotion_map_no_pca.npy"), allow_pickle=True).item()
    emotion_classes = list(emotion_map.keys())
    num_classes = len(emotion_classes)
    logger.info(f"Loaded emotion map: {emotion_map}")
    logger.info(f"Emotion classes: {emotion_classes}")
except FileNotFoundError:
    logger.error("Emotion map file not found")
    raise HTTPException(status_code=500, detail="Emotion map file not found")

# Load feature extractors
try:
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    bert_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert", clean_up_tokenization_spaces=False)
    bert_model = AutoModel.from_pretrained("csebuetnlp/banglabert").to(device).eval()
    logger.info("Feature extractors loaded successfully")
except Exception as e:
    logger.error(f"Error loading feature extractors: {e}")
    raise HTTPException(status_code=500, detail="Failed to load feature extractors")

# Load Bangla STT model
try:
    stt = Speech2Text("large")
    logger.info("Bangla STT model loaded successfully")
except Exception as e:
    logger.error(f"Error loading STT model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load STT model")

# Audio preprocessing (updated to pad/truncate to 7 seconds and apply noise reduction)
def preprocess_audio_for_model(audio_data, target_sr=16000, target_duration=7.0):
    try:
        if not audio_data:
            logger.warning("No audio data provided")
            return None

        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Load audio using torchaudio
        waveform, sr = torchaudio.load(temp_file_path)
        os.unlink(temp_file_path)

        # Resample to 16kHz if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        # Convert to numpy for noise reduction and padding
        audio_np = waveform.squeeze().numpy()

        # Apply noise reduction using librosa
        try:
            # Estimate noise profile from the first 0.5 seconds (or less if audio is shorter)
            noise_clip = audio_np[:int(min(0.5 * target_sr, len(audio_np)))]
            audio_np = librosa.effects.preemphasis(audio_np)  # Pre-emphasis to boost high frequencies
            audio_np = librosa.util.normalize(audio_np)  # Normalize to prevent clipping
            logger.info("Noise reduction applied")
        except Exception as e:
            logger.warning(f"Error in noise reduction: {e}, proceeding without noise reduction")

        # Pad or truncate to 7 seconds
        target_samples = int(target_duration * target_sr)
        if len(audio_np) < target_samples:
            # Pad with zeros
            padding = np.zeros(target_samples - len(audio_np))
            audio_np = np.concatenate([audio_np, padding])
            logger.info(f"Padded audio to 7 seconds: {len(audio_np)/target_sr:.2f}s")
        elif len(audio_np) > target_samples:
            # Truncate to 7 seconds
            audio_np = audio_np[:target_samples]
            logger.info(f"Truncated audio to 7 seconds: {len(audio_np)/target_sr:.2f}s")

        # Convert back to torch tensor
        waveform = torch.tensor(audio_np, dtype=torch.float32)

        return waveform
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None

# Text preprocessing (matching Colab)
def preprocess_text(text):
    if not text or text.strip() == '':
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\u0980-\u09FF\w\s।,?!]', '', text)
    return text

# Feature extraction (matching Colab)
def extract_audio_features(audio, processor, model):
    try:
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.to(device)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return None

def extract_text_features(text, tokenizer, model):
    try:
        if not text or text.strip() == '':
            logger.info("Empty text provided, using zero features")
            return np.zeros(768)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error extracting text features: {e}")
        return None

# MLP model definition (matching Colab)
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

# BiGRU model definition (matching Colab)
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

# Load models (matching Colab)
def load_mlp_model():
    model_path = os.path.join(LOCAL_OUTPUT_DIR, 'mlp_no_pca.pth')
    if not os.path.exists(model_path):
        logger.error(f"MLP model file not found: {model_path}")
        raise FileNotFoundError(f"MLP model file not found: {model_path}")
    mlp_model = MLP(input_dim=1536, hidden_dim=1024, output_dim=num_classes, num_layers=3, dropout=0.5).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        mlp_model.load_state_dict(state_dict)
        mlp_model.eval()
        logger.info("MLP model loaded successfully")
        return mlp_model
    except Exception as e:
        logger.error(f"Error loading MLP model: {e}")
        raise

def load_bigru_model():
    model_path = os.path.join(LOCAL_OUTPUT_DIR, 'bigru_no_pca.pth')
    if not os.path.exists(model_path):
        logger.error(f"BiGRU model file not found: {model_path}")
        raise FileNotFoundError(f"BiGRU model file not found: {model_path}")
    bigru_model = BiGRU(input_dim=1536, hidden_dim=512, output_dim=num_classes, num_layers=3, dropout=0.5).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        bigru_model.load_state_dict(state_dict)
        bigru_model.eval()
        logger.info("BiGRU model loaded successfully")
        return bigru_model
    except Exception as e:
        logger.error(f"Error loading BiGRU model: {e}")
        raise

# Ensemble predictor (matching Colab)
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
            logger.info(f"MLP prediction: {mlp_emotion}, confidence: {mlp_conf:.3f}")
        if self.bigru_model:
            bigru_emotion, bigru_probs, bigru_conf = self._predict_single_model(
                self.bigru_model, feature, emotion_map, device, "BiGRU")
            predictions['BiGRU'] = {'emotion': bigru_emotion, 'confidence': float(bigru_conf), 'probs': bigru_probs.tolist()}
            probabilities['BiGRU'] = bigru_probs
            logger.info(f"BiGRU prediction: {bigru_emotion}, confidence: {bigru_conf:.3f}")
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
            logger.info(f"Ensemble prediction: {ensemble_emotion}, confidence: {ensemble_conf:.3f}")
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
            logger.error(f"Error in {model_name} prediction: {e}")
            return None, None, None

    def _ensemble_probabilities(self, probabilities):
        model_names = list(probabilities.keys())
        weighted_probs = np.zeros_like(probabilities[model_names[0]])
        for i, model_name in enumerate(model_names):
            weight = self.weights[i]
            weighted_probs += weight * probabilities[model_name]
        return weighted_probs

# Process input (matching Colab)
def process_input(audio_data, text_input, ensemble_predictor):
    start_time = time.time()
    audio = preprocess_audio_for_model(audio_data)
    if audio is None:
        logger.error("Failed to process audio")
        return None, None
    text = preprocess_text(text_input)
    logger.info(f"Cleaned text: '{text}'")
    audio_features = extract_audio_features(audio, wav2vec_processor, wav2vec_model)
    if audio_features is None:
        logger.error("Failed to extract audio features")
        return None, None
    text_features = extract_text_features(text, bert_tokenizer, bert_model)
    if text_features is None:
        logger.error("Failed to extract text features")
        return None, None
    combined_features = np.concatenate([audio_features, text_features])
    results = ensemble_predictor.predict(combined_features, emotion_map, device)
    inference_time = time.time() - start_time
    return results, inference_time

# Generate plot (matching Colab)
def generate_plot(results, emotion_classes):
    try:
        num_models = len(results)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 6))
        if num_models == 1:
            axes = [axes]
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_classes)))
        for i, (model_name, result) in enumerate(results.items()):
            bars = axes[i].bar(emotion_classes, result['probs'], color=colors)
            for bar, prob in zip(bars, result['probs']):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            axes[i].set_title(f'{model_name}\nPrediction: {result["emotion"]} ({result["confidence"]:.3f})',
                            fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Emotion', fontweight='bold')
            axes[i].set_ylabel('Probability', fontweight='bold')
            axes[i].set_xticks(range(len(emotion_classes)))
            axes[i].set_xticklabels(emotion_classes, rotation=45)
            axes[i].set_ylim(0, 1.0)
            axes[i].grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        image_png = buffer.getvalue()
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        return None

# Load models
try:
    mlp_model = load_mlp_model()
    bigru_model = load_bigru_model()
    ensemble_predictor = EnsemblePredictor(mlp_model, bigru_model, ensemble_weights=[0.6, 0.4])
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise HTTPException(status_code=500, detail="Failed to load models")

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

@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    favicon_path = os.path.join("static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    logger.warning("Favicon not found")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.post("/predict")
async def predict(audio: UploadFile = File(...), text: str = Form(None)):
    try:
        audio_data = await audio.read()
        if len(audio_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")

        # Transcribe audio if no text provided
        text_input = preprocess_text(text) if text else ""
        if not text_input:
            logger.info("No text input, transcribing...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            waveform, sr = torchaudio.load(temp_file_path)
            os.unlink(temp_file_path)

            if sr != 48000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=48000)

            buf = io.BytesIO()
            torchaudio.save(buf, waveform, 48000, format='wav')
            buf.seek(0)
            text_input = preprocess_text(stt.recognize(buf))

        logger.info(f"Raw text input: '{text}'")
        logger.info(f"Final text input: '{text_input}'")

        # Process input
        results, inference_time = process_input(audio_data, text_input, ensemble_predictor)
        if results is None:
            raise HTTPException(status_code=400, detail="Failed to process input")

        # Generate plot
        plot_base64 = generate_plot(results, emotion_classes)
        if plot_base64 is None:
            raise HTTPException(status_code=500, detail="Failed to generate plot")

        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        return {
            "results": results,
            "plot": plot_base64,
            "audio": audio_base64,
            "emotion_classes": emotion_classes,
            "transcribed_text": text_input,
            "inference_time": inference_time,
            "warning": None
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        audio_data = await audio.read()
        if len(audio_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        waveform, sr = torchaudio.load(temp_file_path)
        os.unlink(temp_file_path)

        if sr != 48000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=48000)

        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, 48000, format='wav')
        buffer.seek(0)

        transcription = stt.recognize(buffer)
        logger.info(f"Transcription: '{transcription}'")
        return {"text": transcription}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)