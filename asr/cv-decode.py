import pandas as pd
import requests
import os
import zipfile
from pathlib import Path
import time
from tqdm import tqdm

# Configuration
DROPBOX_URL = "https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=1"
ZIP_PATH = r"C:\Users\Admin\Downloads\common_voice.zip"
API_URL = "http://localhost:8001/asr"
EXTRACT_PATH = r"C:\Users\Admin\Desktop\Interns\take-home\asr"
CSV_PATH = None  # Will be set after extraction

def download_dataset():
    """Download the Common Voice dataset if not already present"""
    if os.path.exists(ZIP_PATH):
        print(f"Dataset already downloaded at {ZIP_PATH}")
        return True
    
    print("Downloading Common Voice dataset...")
    try:
        response = requests.get(DROPBOX_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(ZIP_PATH, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Dataset downloaded successfully to {ZIP_PATH}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def extract_dataset():
    """Extract the dataset to the ASR folder"""
    global CSV_PATH
    
    if not os.path.exists(ZIP_PATH):
        print(f"ZIP file not found at {ZIP_PATH}")
        return False
    
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        
        # Look for CSV file in multiple possible locations
        possible_csv_paths = [
            os.path.join(EXTRACT_PATH, 'cv-valid-dev.csv'),
            os.path.join(EXTRACT_PATH, 'cv-valid-dev', 'cv-valid-dev.csv'),
            os.path.join(EXTRACT_PATH, 'common_voice', 'cv-valid-dev.csv'),
            os.path.join(EXTRACT_PATH, 'common_voice', 'cv-valid-dev', 'cv-valid-dev.csv')
        ]
        
        # Find the actual CSV file
        for csv_path in possible_csv_paths:
            if os.path.exists(csv_path):
                CSV_PATH = csv_path
                print(f"Found CSV at: {CSV_PATH}")
                return True
        
        # If not found, search recursively
        print("Searching for CSV file...")
        for root, dirs, files in os.walk(EXTRACT_PATH):
            for file in files:
                if file == 'cv-valid-dev.csv':
                    CSV_PATH = os.path.join(root, file)
                    print(f"Found CSV at: {CSV_PATH}")
                    return True
        
        print("Could not find cv-valid-dev.csv file")
        print("Available files and folders:")
        for root, dirs, files in os.walk(EXTRACT_PATH):
            level = root.replace(EXTRACT_PATH, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        return False
        
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def call_asr_api(audio_file_path):
    try:
        if not os.path.exists(audio_file_path):
            return {"error": f"File not found: {audio_file_path}"}
        
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': audio_file}
            response = requests.post(API_URL, files=files, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}: {response.text}"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def process_dataset():
    if not CSV_PATH or not os.path.exists(CSV_PATH):
        print(f"CSV file not found at {CSV_PATH}")
        return
    
    # Load the CSV file
    print(f"Loading CSV file: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} records from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Initialize the generated_text column
    df['generated_text'] = ''
    df['api_duration'] = ''
    df['api_error'] = ''
    
    # Find the audio folder
    csv_dir = os.path.dirname(CSV_PATH)
    audio_folder = os.path.join(csv_dir, 'cv-valid-dev')
    
    if not os.path.exists(audio_folder):
        print(f"Audio folder not found at {audio_folder}")
        return
    
    print(f"Processing audio files from: {audio_folder}")
    
    # Process each file
    processed_count = 0
    error_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
        # Get the filename from the CSV (assuming there's a 'path' column)
        if 'path' in df.columns:
            filename = row['path']
        elif 'filename' in df.columns:
            filename = row['filename']
        else:
            # Try to find filename column
            possible_cols = ['file', 'audio_file', 'mp3_path']
            filename_col = None
            for col in possible_cols:
                if col in df.columns:
                    filename_col = col
                    break
            
            if filename_col:
                filename = row[filename_col]
            else:
                print("Could not find filename column in CSV")
                print("Available columns:", df.columns.tolist())
                return
        
        # Construct full path to audio file
        audio_file_path = os.path.join(audio_folder, filename)
        
        # Call ASR API
        result = call_asr_api(audio_file_path)
        
        if 'error' not in result:
            df.at[index, 'generated_text'] = result.get('transcription', '')
            df.at[index, 'api_duration'] = result.get('duration', '')
            processed_count += 1
        else:
            df.at[index, 'api_error'] = result['error']
            error_count += 1
        
        # Save progress every 100 files
        if (index + 1) % 100 == 0:
            output_path = os.path.join(csv_dir, 'cv-valid-dev-with-transcriptions.csv')
            df.to_csv(output_path, index=False)
            print(f"Progress saved: {processed_count} processed, {error_count} errors")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Save final results
    output_path = os.path.join(csv_dir, 'cv-valid-dev-with-transcriptions.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_path}")

def test_api_connection():
    """Test if the ASR API is running"""
    try:
        health_response = requests.get("http://localhost:8001/health", timeout=5)
        if health_response.status_code == 200:
            print("ASR API is running and healthy")
            return True
        else:
            print(f"ASR API health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot connect to ASR API: {e}")
        print("Make sure your ASR API is running on localhost:8001")
        return False

def main():
    print("Common Voice Dataset Processor")
    print("=" * 40)
    
    # Test API connection first
    if not test_api_connection():
        return
    
    # Check if dataset exists
    if not download_dataset():
        return
    
    # Extract dataset
    if not extract_dataset():
        return
    
    # Process the dataset
    process_dataset()

if __name__ == "__main__":
    main()