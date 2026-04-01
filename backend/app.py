from fastapi import FastAPI, UploadFile, File
import pandas as pd
import tempfile
import os

from model import run_forecast

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Lag-Llama Multi-Context API Running"}


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    prediction_length: int = 28,
    context_lengths: str = "30,60,90"
):
    # Convert context list
    context_lengths = [int(x) for x in context_lengths.split(",")]

    # Save file
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    df = pd.read_csv(temp.name)
    os.unlink(temp.name)

    results = run_forecast(df, prediction_length, context_lengths)

    return {
        "prediction_length": prediction_length,
        "contexts": results
    }