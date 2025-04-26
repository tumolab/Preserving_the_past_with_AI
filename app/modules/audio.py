import tempfile
import scipy.io.wavfile
import torch
from transformers import VitsModel, AutoTokenizer

def load_tts_model():
    model = VitsModel.from_pretrained("facebook/mms-tts-hyw")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hyw")
    return model, tokenizer

def save_audio_waveform(waveform, filename, rate):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    if waveform.ndim > 1:
        waveform = waveform[0]
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    scipy.io.wavfile.write(temp_wav.name, rate=int(rate), data=waveform)
    return temp_wav.name

def handle_tts(text):
    model, tokenizer = load_tts_model()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    return save_audio_waveform(output, "output_audio.wav", model.config.sampling_rate)
