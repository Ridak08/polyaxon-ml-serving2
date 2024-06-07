from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DDOS_CLASS_MAPPING = {0: "BENIGN", 1: "DrDoS_DNS", 2: "Syn", 3: "DrDoS_UDP", 4: "DrDoS_NetBIOS", 5: "DrDoS_NTP", 6: "DrDoS_SNMP", 7: "DrDoS_SSDP"}


def load_model(model_path: str):
    model = open(model_path, "rb")
    return joblib.load(model)


app = FastAPI()
classifier = load_model("./model.joblib")


class DataFeatures(BaseModel):
    source_port: int
    destination_port: int
    protocol: int
    packets: int
    length: float
    fin_flag: float
    syn_flag: int
    rst_flag: int
    psh_flag: int
    ack_flag: int
    urg_flag: float
    cwe_flag: float
    ece_flag: int
    
def get_features(data: DataFeatures) -> np.ndarray:
    return np.array(
        [data.Protocol, data.Flow_Duration, data.Total_Fwd_Packets, data.Total_Backward_Packets, data.Total_Length_of_Fwd_Packets, data.Total_Length_of_Bwd_Packets, data.Inbound],
        ndmin=2,
    )


def predict(features: np.ndarray, proba: bool = False) -> Dict:
    if proba:
        probabilities = {
            k: float(v)
            for k, v in zip(
                DDOS_CLASS_MAPPING.values(), classifier.predict_proba(features)[0]
            )
        }
        return {"probabilities": probabilities}

    prediction = int(classifier.predict(features)[0])
    return {
        "prediction": {"value": prediction, "class": DDOS_CLASS_MAPPING[prediction]}
    }


@app.post("/api/v1/predict")
async def get_prediction(data: DataFeatures):
    features = get_features(data)
    return predict(features)


@app.post("/api/v1/proba")
async def get_probabilities(data: DataFeatures):
    features = get_features(data)
    return predict(features, proba=True)


@app.get("/", response_class=HTMLResponse)
def index():
    return (
        "<p>Hello, This is a REST API used for Polyaxon ML Serving examples!</p>"
        "<p>Click the fullscreen button the get the URL of your serving API!<p/>"
    )
