# main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
import requests
import numpy as np
import librosa
from io import BytesIO

app = FastAPI(
    title="Audio Metrics Service",
    description="Compute average pitch and SNR from a WAV URL",
    version="1.0.0"
)

class MetricsResponse(BaseModel):
    avgPitchHz: float
    snrDb: float

def compute_metrics_from_url(url: str) -> MetricsResponse:
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(400, f"Could not fetch audio: {resp.status_code}")
    y, sr = librosa.load(BytesIO(resp.content), sr=None, mono=True)

    # avg pitch (Hz)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    avg_pitch = float(np.nanmean(f0))

    # SNR (dB)
    intervals = librosa.effects.split(y, top_db=20)
    sig_energy = sum((y[s:e]**2).sum() for s, e in intervals)
    tot_energy = (y**2).sum()
    noise_energy = tot_energy - sig_energy
    snr_db = float(10 * np.log10(sig_energy / (noise_energy + 1e-8)))

    return MetricsResponse(avgPitchHz=avg_pitch, snrDb=snr_db)

@app.get(
    "/api/metrics",
    response_model=MetricsResponse,
    summary="Compute audio metrics",
    description="Given a publicly accessible WAV file URL, returns average pitch (Hz) and signal-to-noise ratio (dB)."
)
async def api_metrics(
    url: HttpUrl = Query(
        ...,
        description="Direct URL to a mono WAV file"
    )
):
    """
    JSON endpoint:

    **GET** `/api/metrics?url=<your-wav-url>`

    Returns:
    ```json
    {
      "avgPitchHz": 180.7,
      "snrDb": 22.3
    }
    ```
    """
    return compute_metrics_from_url(str(url))

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return """
    <html>
      <head><title>Audio Metrics Service</title></head>
      <body style="font-family:sans-serif;padding:2rem;">
        <h1>Audio Metrics Service</h1>
        <p>
          👉 To compute metrics, go to
          <a href="/docs" target="_blank">/docs</a>
          or call <code>/api/metrics?url=...</code>
        </p>
      </body>
    </html>
    """

# Run with:
# uvicorn main:app --reload --port 8000
