from flask import Flask, request, jsonify
import torch
import librosa
import tempfile
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Model configuration
MODEL_NAME = "facebook/wav2vec2-large-960h"

# Initialize model and processor as None
processor = None
model = None

def initialize_model():
    """Initialize the model and processor"""
    global processor, model
    try:
        print("Loading model and processor...")
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/asr', methods=['POST'])
def transcribe_audio():
    global processor, model
    
    try:
        # Initialize model if not already loaded
        if processor is None or model is None:
            if not initialize_model():
                return jsonify({"error": "Failed to load model"}), 500
            
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Create temporary file to save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and process audio file (resample to 16kHz)
            audio_input, sample_rate = librosa.load(tmp_file_path, sr=16000)
            
            # Calculate duration
            duration = len(audio_input) / sample_rate
            duration_str = f"{duration:.1f}"
            
            # Process audio with the processor
            input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True).input_values
            
            # Perform inference
            with torch.no_grad():
                logits = model(input_values).logits
            
            # Decode the logits to get transcription
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return jsonify({
                "transcription": transcription,
                "duration": duration_str
            })
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "ASR API is running!",
        "endpoints": {
            "transcribe": "POST /asr (with file parameter)",
            "health": "GET /health"
        },
        "usage": "curl -F 'file=@yourfile.mp3' http://localhost:8001/asr"
    })

@app.route('/health', methods=['GET'])
def health_check():
    global processor, model
    status = "healthy" if processor is not None and model is not None else "model_not_loaded"
    return jsonify({"status": status, "model": MODEL_NAME})

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs('/tmp/asr_uploads', exist_ok=True)
    
    # Load model at startup
    print("Starting ASR API server...")
    initialize_model()
    app.run(host='0.0.0.0', port=8001, debug=False)