from fastapi import FastAPI,UploadFile,File
from numpy.core.defchararray import index
from numpy.lib.type_check import imag
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
MODEL= tf.keras.models.load_model(r"C:\Users\KIIT\Desktop\Apps\Project\Potato Disease Detector\saved_model\1")
CLASS_NAMES =["Early Bright","Late Blight","HEALTHY"]



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
    predictions= MODEL.predict(img_batch)  
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence= np.max(predictions[0])
    print(predicted_class,confidence)
    return{
        "class": predicted_class,
        "confidence":float(confidence)}
    


if __name__== "__main__":
    uvicorn.run(app,host='localhost',port=8000)