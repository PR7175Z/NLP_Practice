from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import traceback
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# vectorizer = joblib.load('vectorizer.pkl')
loaded_model = tf.keras.models.load_model('sentiment_model.keras') 

class Feature(BaseModel):
    text: str

@app.post('/predict')
def predict(data: Feature):
    try:
        # input_vect = vectorizer(tf.convert_to_tensor([data.text]))
        input_tensor = tf.convert_to_tensor([data.text])
        prediction = loaded_model.predict(input_tensor)
        return {"prediction": prediction[0].tolist()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))