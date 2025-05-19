# main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
import wave
import numpy as np
import struct
from io import BytesIO
import requests

app = FastAPI(
    title="Lightweight Audio Metrics",
    description="Compute avg pitch and SNR without librosa",
    version="1.0.0"
)

class PitchResponse(BaseModel):
    avgPitchHz: float

class SnrResponse(BaseModel):
    snrDb: float

def fetch_wave_bytes(url: str):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(400, f"Could not fetch audio: {resp.status_code}")
    return BytesIO(resp.content)

def read_wave(buf):
    """Return (samples as float32 numpy array, sample_rate)."""
    with wave.open(buf, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        chans = wf.getnchannels()
        width = wf.getsampwidth()
    # unpack to ints
    fmt = {1:'b', 2:'h', 4:'i'}[width]
    count = n * chans
    ints = np.array(struct.unpack(f"<{count}{fmt}", raw), dtype=np.float32)
    # if stereo, just take first channel
    if chans > 1:
        ints = ints.reshape(-1, chans)[:,0]
    # normalize to -1..1
    ints /= float(2**(8*width-1))
    return ints, sr

def compute_avg_pitch(samples, sr):
    frame_ms = 30
    hop_ms   = 15
    frame_len = int(sr * frame_ms/1000)
    hop_len   = int(sr * hop_ms/1000)
    freqs = []
    window = np.hamming(frame_len)
    for start in range(0, len(samples)-frame_len, hop_len):
        frame = samples[start:start+frame_len] * window
        # FFT
        spectrum = np.abs(np.fft.rfft(frame))
        if spectrum.sum() < 1e-6:
            continue
        peak = np.argmax(spectrum)
        freq = peak * sr / frame_len
        if 50 < freq < 500:  # consider human voice band
            freqs.append(freq)
    if not freqs:
        return 0.0
    return float(np.mean(freqs))

def compute_snr(samples, sr):
    frame_ms = 30
    hop_ms   = 15
    frame_len = int(sr * frame_ms/1000)
    hop_len   = int(sr * hop_ms/1000)
    energies = []
    for start in range(0, len(samples)-frame_len, hop_len):
        frame = samples[start:start+frame_len]
        energies.append(np.sqrt(np.mean(frame**2)))
    energies = np.array(energies)
    if len(energies) < 10:
        return 0.0
    # sort
    sorted_e = np.sort(energies)
    cutoff = max(1, len(sorted_e)//10)
    noise_rms  = np.mean(sorted_e[:cutoff])
    signal_rms = np.mean(sorted_e[-cutoff:])
    # avoid divide by zero
    if noise_rms < 1e-8:
        return float('inf')
    return float(20 * np.log10(signal_rms / noise_rms))

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return """
    <html><body>
      <h1>Lightweight Audio Metrics</h1>
      <p>Endpoints:</p>
      <ul>
        <li>GET /api/avg-pitch?url=&lt;wav-url&gt;</li>
        <li>GET /api/snr?url=&lt;wav-url&gt;</li>
      </ul>
      <p>Interactive docs: <a href="/docs">/docs</a></p>
    </body></html>
    """

@app.get("/api/avg-pitch", response_model=PitchResponse, summary="Avg Pitch")
async def api_avg_pitch(url: HttpUrl = Query(...)):
    buf = fetch_wave_bytes(str(url))
    samples, sr = read_wave(buf)
    return PitchResponse(avgPitchHz=compute_avg_pitch(samples, sr))

@app.get("/api/snr", response_model=SnrResponse, summary="SNR")
async def api_snr(url: HttpUrl = Query(...)):
    buf = fetch_wave_bytes(str(url))
    samples, sr = read_wave(buf)
    return SnrResponse(snrDb=compute_snr(samples, sr))

# Run:
# uvicorn main:app --reload --port 8000
