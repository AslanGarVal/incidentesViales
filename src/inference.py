from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn  # for local testing

from xgboost import XGBRegressor
import numpy as np
import os

class PredictionRequest(BaseModel):
    incidente_5: float
    incidente_4: float
    incidente_3: float
    incidente_2: float
    incidente_1: float
    day: float
    month: float
    day_of_week: float
    hour: float
    tgt_enc_deleg: float


# declare API instance
app = FastAPI()

# load model
#path_to_model = 'C:\\Users\\Aslan García\\PycharmProjects\\incidentesViales\\src'
#model_path = os.path.join(path_to_model, 'car_incidents_model.json')
model = XGBRegressor()
model.load_model('car_incidents_model.json')

@app.post('/predict')
def predict(data: PredictionRequest):
    features = np.array([data.incidente_5, data.incidente_4, data.incidente_3,
                data.incidente_2, data.incidente_1, data.day, data.month, data.day_of_week,
                data.hour, data.tgt_enc_deleg]).reshape(1, 10)
    prediction = model.predict(features).tolist()
    return {'predicted transit incidents': prediction}

