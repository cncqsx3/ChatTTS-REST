# Standard library imports
import logging
import os
import uuid
from contextlib import asynccontextmanager
from html import escape
from typing import List, Optional # Optional can be removed if not used

# Third-party imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import torch
import torchaudio

# Application-specific imports
import ChatTTS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
logger = logging.getLogger(__name__) # Define logger globally


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Initializing ChatTTS model...")
    try:
        # Check for GPU availability and log it
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("PyTorch CUDA is not available. Using CPU for ChatTTS.")

        app.state.chattts_model = ChatTTS.Chat()
        # chat.load() is often called implicitly or can be called explicitly.
        # The default behavior of ChatTTS.Chat() should handle loading.
        # If specific loading like chat.load(compile=True) is needed, it can be added here.
        # For now, direct initialization is assumed to be sufficient based on docs.
        logger.info("ChatTTS model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ChatTTS model: {e}")
        app.state.chattts_model = None # Indicate failure
    yield
    # Clean up the ML model and release the resources
    logger.info("Cleaning up ChatTTS model (if any)...")
    if hasattr(app.state, 'chattts_model') and app.state.chattts_model is not None:
        # Add cleanup code here if ChatTTS library provides explicit deallocation methods
        # For now, just dereference
        app.state.chattts_model = None
        logger.info("ChatTTS model resources released (simulated).")

app = FastAPI(lifespan=lifespan)

# Mount static files directory
app.mount("/static/wavs", StaticFiles(directory="static/wavs"), name="static_wavs")

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


@app.post("/tts", response_model=TTSResponse) # Add response_model back
async def create_tts_request(request: Request, request_data: TTSRequest):
    # logger = logging.getLogger(__name__) # Global logger is already defined and accessible
    global logger # Make it explicit that we are using the global logger

    chat = request.app.state.chattts_model
    if chat is None:
        logger.error("ChatTTS model not available.")
        return JSONResponse(status_code=503, content={"code": 1, "msg": "error"})

    # Ensure text is a string before escaping, default to empty string if not
    text_to_sanitize = request_data.text if isinstance(request_data.text, str) else ""
    sanitized_text = escape(text_to_sanitize)
    logger.info(f"Received TTS request: voice='{request_data.voice}', temp='{request_data.temperature}', text='{sanitized_text}'")

    params_infer = {
        "text": [sanitized_text], # Must be a list
        "use_prompt": True if request_data.prompt else False,
        "prompt": request_data.prompt,
        "temperature": request_data.temperature,
        "top_P": request_data.top_p,
        "top_K": request_data.top_k,
        "skip_refine": bool(request_data.skip_refine),
        "lang": "EN"
    }
    
    voice_seed = None
    rand_spk_arg = None
    if request_data.voice and request_data.voice.lower() not in ['default', '']:
        try:
            voice_seed = int(request_data.voice)
        except ValueError:
            logger.warning(f"Voice ID '{request_data.voice}' is not a valid integer. Proceeding without specific speaker embedding.")
    
    if voice_seed is not None:
        try:
            rand_spk_arg = chat.sample_random_speaker(seed=voice_seed)
            logger.info(f"Using voice ID {voice_seed} to generate speaker embedding.")
        except Exception as e:
            logger.error(f"Error sampling random speaker with seed {voice_seed}: {e}", exc_info=True)
            # Proceeding without speaker embedding if sampling fails

    if rand_spk_arg is not None:
        params_infer["speaker_prompt"] = rand_spk_arg # Corrected key

    # Construct a string for logging that omits the full speaker_prompt if it's large
    log_params = {k: v for k, v in params_infer.items() if k != "speaker_prompt"}
    if "speaker_prompt" in params_infer and params_infer["speaker_prompt"] is not None:
        log_params["speaker_prompt_shape"] = params_infer["speaker_prompt"].shape # Log shape
    logger.info(f"Calling ChatTTS.infer with params: {log_params}")

    try:
        wavs = chat.infer(**params_infer)
    except Exception as e:
        logger.error(f"Error during ChatTTS inference: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})

    output_dir = "static/wavs"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating output directory {output_dir}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})

    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(output_dir, filename)

    try:
        # wavs is a list of tensors. Save the first one.
        if not wavs or not isinstance(wavs, list) or not len(wavs[0]): # Check from prompt
             logger.error(f"ChatTTS inference returned empty or invalid result: {wavs}")
             return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})
        torchaudio.save(filepath, wavs[0], 24000)  # ChatTTS default sample rate is 24k
        logger.info(f"Audio saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving audio to {filepath}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"code": 1, "msg": "error"})
    
    # Construct the full URL for the audio file
    scheme = request.url.scheme
    hostname = request.url.hostname
    port = request.url.port
    
    if port:
        full_url = f"{scheme}://{hostname}:{port}/static/wavs/{filename}"
    else: # Port might be None for standard 80/443
        full_url = f"{scheme}://{hostname}/static/wavs/{filename}"
        
    logger.info(f"Constructed audio URL: {full_url}")

    return TTSResponse(
        code=0,
        msg="ok",
        audio_files=[AudioFile(filename=filename, url=full_url)]
    )


if __name__ == "__main__":
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=HOST, port=PORT)
