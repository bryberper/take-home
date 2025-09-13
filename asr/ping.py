import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import Dict
import torch
from transformers import pipeline

# Initialize the FastAPI application
app = FastAPI(
    title="ASR Microservice",
    description="A microservice for transcribing audio files using a pre-trained ASR model.",
)

# Initialize the ASR pipeline
# NOTE: The model will be downloaded automatically the first time the app runs.
# This may take a while.
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error initializing ASR pipeline: {e}")
    asr_pipeline = None

@app.get("/ping", tags=["Health Check"])
def ping() -> Dict[str, str]:
    """
    A simple health check endpoint that returns "pong".
    """
    return {"response": "pong"}

@app.post("/transcribe/", tags=["ASR"])
async def transcribe_audio(audio_file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Transcribes an audio file uploaded as multipart form data.
    """
    if asr_pipeline is None:
        return {"error": "ASR model is not loaded. Please check the logs for more details."}

    try:
        # Read the audio file into memory
        audio_data = await audio_file.read()
        
        # Transcribe the audio
        transcribed_text = asr_pipeline(audio_data)["text"]
        
        return {"transcribed_text": transcribed_text}
    except Exception as e:
        # It's good practice to catch and return a more user-friendly error
        return {"error": f"An error occurred during transcription: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)