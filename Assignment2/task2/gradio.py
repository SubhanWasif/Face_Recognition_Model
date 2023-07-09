

import gradio as gr
# import tensorflow as tf
import gradio.inputs as inp
import gradio.outputs as out
from PIL import Image
import numpy as np
import pickle
# Load the pre-trained TensorFlow model
# model = tf.keras.models.load_model("path/to/model")
with open('/home/schneider/Desktop/ML/Assignment3/task2/LR.pkl', 'rb') as f:
    LR = pickle.load(f)

# Define a function that takes an input image and returns the predicted label and probability
def predict_image(input_image):
    image = input_image.copy()
    image = processimage(image)
    pred = LR.predict(image)
    return pred
    


def processimage(image):
    new_image=[]
    new_size = (32,32,3)
    resized_img = np.resize(image,new_size)
    img_ = np.divide(resized_img, 255)
    img = img_.reshape(3072)
    new_image.append(img)
    return new_image

    # Return the predicted label and probability as a dictionary
    # return {"label": predicted_label, "probability": probability}

# Define the input component of the GUI
input_image = inp.Image(shape=(200,200))


# Define the output component of the GUI
output_label = out.Label(num_top_classes=1)

# Define the interface for the GUI
interface = gr.Interface(fn=predict_image, inputs='text', outputs=output_label, title="Image Classifier")

# Launch the interface
interface.launch()
