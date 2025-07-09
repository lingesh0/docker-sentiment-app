from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Dummy logic: positive if 'love' in text, else negative
    label = "positive" if "love" in request.text.lower() else "negative"
    score = 0.99 if label == "positive" else 0.85
    return PredictResponse(label=label, score=score) 