import pickle
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

class PredictionInput(BaseModel):
    features: list[float] 

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_configs = {
        "diabetes": 'trained_model/lgb_model_diabetes.pkl',
        "typhoid": 'trained_model/lgb_model_typhoid.pkl' 
    }
    
    for model_name, path in model_configs.items():
        try:
            with open(path, 'rb') as file:
                ml_models[model_name] = pickle.load(file)
            print(f"Model '{model_name}' loaded successfully from {path}")
        except FileNotFoundError:
            print(f"Error: Model file for '{model_name}' not found at {path}")
            ml_models[model_name] = None
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            ml_models[model_name] = None
    
    yield 
    
    ml_models.clear()

app = FastAPI(title="Multi-Disease Prediction API", lifespan=lifespan)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _predict(data: PredictionInput, model_name: str, required_features: int, threshold: float):
    
    model = ml_models.get(model_name)
    
    if not model:
        raise HTTPException(status_code=500, detail=f"Model '{model_name}' is not loaded.")

    try:
        input_array = np.array(data.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")

    if input_array.shape[1] != required_features:
          raise HTTPException(
              status_code=400, 
              detail=f"Model '{model_name}' expects {required_features} features, got {input_array.shape[1]}"
          )

    try:
        probs = model.predict_proba(input_array)
        prob_positive = probs[0][1] 
    except AttributeError:
        raise HTTPException(status_code=500, detail="Loaded model does not support probability prediction.")

    prediction = 1 if prob_positive >= threshold else 0
    label = f"Positive ({model_name.capitalize()})" if prediction == 1 else f"Negative (No {model_name.capitalize()})"

    return {
        "prediction_class": prediction,
        "prediction_label": label,
        "probability_positive": float(prob_positive),
        "threshold_used": threshold,
        "is_above_threshold": bool(prob_positive >= threshold)
    }


@app.post("/predict/diabetes")
async def predict_diabetes(data: PredictionInput):
    REQUIRED_FEATURES_DIABETES = 5
    THRESHOLD_DIABETES = 0.8
    return _predict(data, "diabetes", REQUIRED_FEATURES_DIABETES, THRESHOLD_DIABETES)


@app.post("/predict/typhoid")
async def predict_typhoid(data: PredictionInput):
    REQUIRED_FEATURES_TYPHOID = 5
    THRESHOLD_TYPHOID = 0.75 
    return _predict(data, "typhoid", REQUIRED_FEATURES_TYPHOID, THRESHOLD_TYPHOID)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)