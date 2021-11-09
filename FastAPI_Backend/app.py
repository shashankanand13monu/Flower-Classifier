from fastapi import FastAPI,UploadFile,File
from numpy.core.defchararray import index
from numpy.lib.type_check import imag
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://fronend-web-app.herokuapp.com/",
    "https://frontend-web-app.herokuapp.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL= tf.keras.models.load_model("./flower_model.h5")
CLASS_NAMES =["Daisy","Dandelion","Rose","SunFlower","Tulip"]



@app.get("/ping")
async def ping():
    return "Sup Brother"

def read_file_as_image(data) -> np.ndarray:
    image= np.array(Image.open(BytesIO(data)))
    return image
    
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

@app.post("/predict")
async def predict(file: UploadFile= File(...)):
    image= read_file_as_image(await file.read())
    img_batch= np.expand_dims(image,axis=0)
    predictions= MODEL.predict(img_batch)  
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence= np.max(predictions[0])
    print(predicted_class,confidence)
    return{
        "class": predicted_class,
        "confidence":float(confidence)}
    


if __name__== "__main__":
    uvicorn.run(app,host='localhost',port=8000)