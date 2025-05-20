# main.py

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
import requests, numpy as np, librosa
from io import BytesIO

app = FastAPI(
    title="Audio Metrics Service",
    description="Compute average pitch and SNR from a WAV URL",
    version="1.1.1"
)

class PitchResponse(BaseModel):
    avgPitchHz: float

class SnrResponse(BaseModel):
    snrDb: float

def fetch_audio(url: str):
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(400, f"Could not fetch audio: {resp.status_code}")
    return librosa.load(BytesIO(resp.content), sr=None, mono=True)

def calculate_snr(y: np.ndarray) -> float:
    intervals = librosa.effects.split(y, top_db=20)
    if not intervals:
        return 0.0
    sig_energy = sum((y[s:e]**2).sum() for s, e in intervals)
    tot_energy = (y**2).sum()
    noise_energy = tot_energy - sig_energy
    if noise_energy <= 0 or sig_energy <= 0:
        return 0.0
    return float(10 * np.log10(sig_energy / noise_energy))

def calculate_avg_pitch(y: np.ndarray) -> float:
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    # filter out NaNs
    valid = f0[~np.isnan(f0)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    # Catch everything and return JSON
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"error": str(exc)}
    )

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return """
    <html><body>
      <h1>Audio Metrics Service</h1>
      <p>Try the endpoints in <a href="/docs">/docs</a></p>
    </body></html>
    """

@app.get("/api/avg-pitch", response_model=PitchResponse, summary="Compute average pitch")
async def api_avg_pitch(url: HttpUrl = Query(..., description="WAV file URL")):
    y, _ = fetch_audio(str(url))
    return PitchResponse(avgPitchHz=calculate_avg_pitch(y))

@app.get("/api/snr", response_model=SnrResponse, summary="Compute SNR")
async def api_snr(url: HttpUrl = Query(..., description="WAV file URL")):
    y, _ = fetch_audio(str(url))
    return SnrResponse(snrDb=calculate_snr(y))
