from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from challenge.model import DelayModel

app = FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


# Load allowed categorical values once at startup
_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"
try:
    _DF = pd.read_csv(_DATA_PATH)
    ALLOWED_OPERAS = set(_DF["OPERA"].dropna().unique())
    ALLOWED_TIPOVUELO = set(_DF["TIPOVUELO"].dropna().unique())
except Exception:
    ALLOWED_OPERAS = None
    ALLOWED_TIPOVUELO = {"I", "N"}


# Shared model instance
MODEL = DelayModel()


@app.post("/predict", status_code=200)
async def post_predict(payload: Dict[str, Any]) -> Dict[str, List[int]]:
    # Basic payload validation
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    flights = payload.get("flights")
    if not isinstance(flights, list) or len(flights) == 0:
        raise HTTPException(status_code=400, detail="'flights' must be a non-empty list")

    # Validate each flight record
    for f in flights:
        if not isinstance(f, dict):
            raise HTTPException(status_code=400)

        # MES must be integer between 1 and 12
        mes = f.get("MES")
        if not isinstance(mes, int) or not (1 <= mes <= 12):
            raise HTTPException(status_code=400)

        # TIPOVUELO must be known
        tip = f.get("TIPOVUELO")
        if tip is None or (ALLOWED_TIPOVUELO is not None and tip not in ALLOWED_TIPOVUELO):
            raise HTTPException(status_code=400)

        # OPERA must be known
        opera = f.get("OPERA")
        if opera is None or (ALLOWED_OPERAS is not None and opera not in ALLOWED_OPERAS):
            raise HTTPException(status_code=400)

    # Convert to DataFrame, preprocess and predict
    df = pd.DataFrame(flights)
    features = MODEL.preprocess(df)
    preds = await run_in_threadpool(MODEL.predict, features)

    # Ensure JSON-serializable list of ints
    try:
        preds_list = [int(x) for x in preds]
    except Exception:
        preds_list = list(preds)

    return {"predict": preds_list}