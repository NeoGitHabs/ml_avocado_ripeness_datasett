from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import uvicorn

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_tree_AvocadoRipenessDataset.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_AvocadoRipenessDataset.pkl')

avocado_app = FastAPI()


class AvocadoSchema(BaseModel):
    firmness: float
    hue: float
    saturation: float
    brightness: float
    color_category: str
    sound_db: float
    weight_g: float
    size_cm3: float


@avocado_app.post('/predict')
async def predict(avocado: AvocadoSchema):
    color = avocado.color_category
    color_0_1 = [
        1 if color == "dark green" else 0,
        1 if color == "green" else 0,
        1 if color == "purple" else 0
    ]

    numerical_features = [
        avocado.firmness,
        avocado.hue,
        avocado.saturation,
        avocado.brightness,
        avocado.sound_db,
        avocado.weight_g,
        avocado.size_cm3
    ]

    features = numerical_features + color_0_1
    scaled = scaler.transform([features])

    pred = str(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0][1]
    print(f"Probability: {proba}")

    return {'status': pred}


if __name__ == '__main__':
    uvicorn.run("main:avocado_app", host="127.0.0.1", port=8000, reload=True)