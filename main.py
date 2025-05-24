from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import logging
import os
import shutil
from pydantic import BaseModel
from html import escape
from typing import List, Optional
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    prompt: str
    voice: str
    temperature: float
    top_p: float
    top_k: int
    skip_refine: int
    custom_voice: int


class AudioFile(BaseModel):
    filename: str
    url: str


class TTSResponse(BaseModel):
    code: int
    msg: str
    audio_files: List[AudioFile]


@app.post("/tts", response_model=TTSResponse) # Added response_model
async def create_tts_request(request_data: TTSRequest):
    logger = logging.getLogger(__name__)
    
    # Ensure text is a string before escaping, default to empty string if not
    text_to_sanitize = request_data.text if isinstance(request_data.text, str) else ""
    sanitized_text = escape(text_to_sanitize)
    logger.info(f"Received TTS request: voice='{request_data.voice}', temp='{request_data.temperature}', text='{sanitized_text}'")

    chat_tts_payload = {
        "text": sanitized_text,
        "prompt": request_data.prompt,
        "voice": request_data.voice,
        "temperature": request_data.temperature,
        "top_p": request_data.top_p,
        "top_k": request_data.top_k,
        "skip_refine": request_data.skip_refine,
        "custom_voice": request_data.custom_voice,
    }

    logger.info(f"Sending payload to ChatTTS: {chat_tts_payload}")
    chat_tts_url = "http://127.0.0.1:9966/tts"

    try:
        response = requests.post(chat_tts_url, json=chat_tts_payload)
        logger.info(f"ChatTTS response status code: {response.status_code}")
        # Limiting log of content to first 500 chars for brevity
        logger.debug(f"ChatTTS response content: {response.text[:500]}")

        if response.status_code == 200:
            try:
                response_json = response.json() # Attempt to parse JSON
                logger.info(f"ChatTTS request successful, JSON response received: {response_json}")

                filename = response_json.get('filename')
                if not filename or not isinstance(filename, str):
                    logger.error(f"ChatTTS response missing or invalid filename: {filename}")
                    return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})

                target_dir = "E:/python/chattts/static/wavs/"
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    logger.info(f"Ensured directory exists: {target_dir}")
                except OSError as e:
                    logger.error(f"Error creating directory {target_dir}: {e}")
                    return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})

                source_audio_url = f"http://127.0.0.1:9966/static/wavs/{filename}"
                logger.info(f"Attempting to download audio from: {source_audio_url}")

                try:
                    audio_response = requests.get(source_audio_url, stream=True)
                    audio_response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading audio from {source_audio_url}: {e}")
                    return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})

                target_file_path = os.path.join(target_dir, filename)
                
                try:
                    with open(target_file_path, 'wb') as f:
                        shutil.copyfileobj(audio_response.raw, f)
                    logger.info(f"Successfully saved audio file to {target_file_path}")
                except IOError as e:
                    logger.error(f"Error saving audio file to {target_file_path}: {e}")
                    return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})
                
                return TTSResponse(code=0, msg="ok", audio_files=[AudioFile(filename=filename, url=source_audio_url)])

            except ValueError as e: # For response.json() failing
                logger.error(f"Error parsing ChatTTS JSON response: {e}")
                return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})
        else: # status_code != 200
            logger.error(f"Error from ChatTTS service. Status: {response.status_code}, Body: {response.text[:500]}")
            return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling ChatTTS: {e}")
        return JSONResponse(status_code=502, content={"code": 1, "msg": "error"})
    # A general except Exception is not added here to stick to specified error handling.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
