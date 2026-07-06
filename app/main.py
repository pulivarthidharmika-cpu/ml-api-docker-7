from fastapi import FastAPI
from app.schemas import WineInput
from app.predict import predict_wine

app = FastAPI(
    title="Wine Classification API",
    description="A FastAPI application to classify wine using a Machine Learning model.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to the Wine Classification API!",
        "status": "API is running successfully."
    }


@app.get("/health")
def health_check():
    return {
        "status": "Healthy"
    }


@app.post("/predict")
def predict(data: WineInput):
    prediction = predict_wine(data)

    classes = {
        0: "Class 0",
        1: "Class 1",
        2: "Class 2"
    }

    return {
        "prediction": prediction,
        "wine_class": classes[prediction]
    }