from fastapi import FastAPI, HTTPException
from schema import Endpoint, Query
from predictor import load_index, predict_from_text


# --- FastAPI Application Instance ---
app = FastAPI(
    title="Konie API endpoint predictor transformer",
    description="Word embedding transformer for predicting the possible endpoint from Konie API for a given insert query",
    version="0.1.0",
)

@app.get("/", tags=["Root"])
def read_root():
    """
    **Healthcheck Path**
    
    Healthcheck
    """
    return {"status": "ok"}

# ---
@app.post("/predict", response_model=Endpoint, tags=["Transformer"])
def predict(query: Query):
    """
    **Predict the endpoint**
    """
    try:
        pred_ep = predict_from_text(query.prompt)
        return Endpoint(prompt=query.prompt, schemajson="", endpoint=pred_ep)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas predykcji: {e}")

# ---
@app.post("/recreate", status_code=201, tags=["Transformer"])
def recreate():
    try:
        load_index(force_rebuild=True)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nie udało się przebudować indeksu: {e}")

# if __name__ == "__main__":
#     import uvicorn
#     load_index(force_rebuild=False)
#     uvicorn.run(app, host="0.0.0.0", port=8000)