from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import requests, numpy as np, librosa
from io import BytesIO

app = FastAPI(title="SNR Service")

class SnrResponse(BaseModel):
    snrDb: float

def fetch_audio(url: str):
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise HTTPException(400, "Fetch failed")
    return librosa.load(BytesIO(r.content), sr=None, mono=True)

def calc_snr(y):
    intervals = librosa.effects.split(y, top_db=20)
    if not len(intervals): 
        return 0.0
    sig = sum((y[s:e]**2).sum() for s,e in intervals)
    tot = (y**2).sum()
    noise = tot - sig
    if sig<=0 or noise<=0:
        return 0.0
    return float(10*np.log10(sig/noise))

@app.get("/api/snr", response_model=SnrResponse)
async def snr(url: HttpUrl = Query(...)):
    y, _ = fetch_audio(str(url))
    return SnrResponse(snrDb=calc_snr(y))
