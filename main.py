# main.py

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
import requests, numpy as np, librosa
from io import BytesIO

app = FastAPI(
    title="Audio Metrics Service",
    description="Compute average pitch and SNR from a WAV URL",
    version="1.1.2"
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

def calculate_avg_pitch(y: np.ndarray) -> float:
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    valid = f0[~np.isnan(f0)]
    return float(np.mean(valid)) if valid.size else 0.0

def calculate_snr(y: np.ndarray) -> float:
    intervals = librosa.effects.split(y, top_db=20)
    if len(intervals) == 0:
        return 0.0
    sig = sum((y[s:e]**2).sum() for s, e in intervals)
    tot = (y**2).sum()
    noise = tot - sig
    if sig <= 0 or noise <= 0:
        return 0.0
    return float(10 * np.log10(sig / noise))

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
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

# To run:
# python -m uvicorn main:app --reload --port 8000
