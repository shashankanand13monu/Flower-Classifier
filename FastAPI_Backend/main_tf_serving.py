from fastapi import FastAPI,UploadFile,File
from numpy.core.defchararray import index
from numpy.lib.type_check import imag
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model/predict"
# MODEL= tf.keras.models.load_model(r"C:\Users\KIIT\Desktop\Apps\Project\Potato Disease Detector\saved_model\1")
CLASS_NAMES =["Early Bright","HEALTHY","Late Blight"]



@app.get("/ping")
async def ping():
    return "Sup Brother"

def read_file_as_image(data) -> np.ndarray:
    image= np.array(Image.open(BytesIO(data)))
    return image
    

@app.post("/predict")
async def predict(file: UploadFile= File(...)):
    image= read_file_as_image(await file.read())
    img_batch= np.expand_dims(image,axis=0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response= requests.post(endpoint, json=json_data)
    prediction= response.json()["predictions"][0]
    predicted_class= CLASS_NAMES[np.argmax(prediction)]
    confidence= np.max(prediction)
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

    
   


if __name__== "__main__":
    uvicorn.run(app,host='localhost',port=8000)