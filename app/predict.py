import pandas as pd

from app.model import model

def predict_wine(data):
    sample = pd.DataFrame([{
        "alcohol": data.alcohol,
        "malic_acid": data.malic_acid,
        "ash": data.ash,
        "alcalinity_of_ash": data.alcalinity_of_ash,
        "magnesium": data.magnesium,
        "total_phenols": data.total_phenols,
        "flavanoids": data.flavanoids,
        "nonflavanoid_phenols": data.nonflavanoid_phenols,
        "proanthocyanins": data.proanthocyanins,
        "color_intensity": data.color_intensity,
        "hue": data.hue,
        "od280_od315": data.od280_od315,
        "proline": data.proline
    }])

    prediction = model.predict(sample)

    return int(prediction[0])