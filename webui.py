import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import platform

from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")

    # Determine appropriate device based on platform and availability
    if platform.system() == "Darwin":
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    model = SparkTTS(model_dir, device)
    return model


def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )

        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")

    return save_path


if __name__ == "__main__":
    spark_base=initialize_model()
    ref_text="""Hello! My name is Clara, your personal AI Nurse. It’s so nice to meet you, Emma. I’m 
                here to welcome you to our new Medicaid program. I understand that navigating health care can 
                sometimes feel overwhelming, but I’m here to make things easier and clearer for you, every step of the 
                way. """
    ref_speech="ref/voice-message1-clara-intro.wav"
    prompt="Hi, how are you doing?"

    audio, sr = sf.read(ref_speech)
    print(f"Loaded reference audio: shape={audio.shape}, sample_rate={sr}")
    run_tts(text=prompt, model=spark_base,prompt_text=ref_text,prompt_speech=ref_speech)

