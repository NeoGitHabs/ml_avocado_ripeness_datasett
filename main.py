from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn


model = joblib.load('model_tree.pkl')
scaler = joblib.load('scaler.pkl')

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

type_color = ['dark green', 'green', 'purple']


@avocado_app.post('/predict')
async def predict(avocado: AvocadoSchema):
    avocado_dict = avocado.dict()

    color = avocado_dict.pop('color_category')

    color_0_1 = [
        1 if color == "dark green" else 0,
        1 if color == "green" else 0,
        1 if color == "purple" else 0
    ]

    features = list(avocado_dict.values()) + color_0_1
    scaled = scaler.transform([features])

    pred = model.predict(scaled)[0]

    proba = model.predict_proba(scaled)[0][1]
    print(proba)

    return {'approved': pred}


if __name__ == '__main__':
    uvicorn.run("main:avocado_app", host="127.0.0.1", port=8000, reload=True)
