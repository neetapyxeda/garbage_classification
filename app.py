import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.resnet import preprocess_input
from PIL import Image
import pickle
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


# constants
IMG_SIZE = (224, 224)
IMG_ADDRESS = "https://t4.ftcdn.net/jpg/03/56/69/55/360_F_356695553_DGCg4F6KpySgbyzQBIBb2FmOIH6Vj51m.jpg"
IMAGE_NAME = "user_image.png"
CLASS_LABEL = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_LABEL.sort()


@st.cache_resource
def get_ConvNeXtXLarge_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.ConvNeXtXLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions


# get the featurization model
ConvNeXtXLarge_featurized_model = get_ConvNeXtXLarge_model()
# load ultrasound image
classification_model = load_sklearn_models("MLP_Epochs_40_LearningRate_0.0001.pkl")


# web app

# title
st.title("Garbage Classification")
# image
st.image(IMG_ADDRESS, caption = "Garbage Classification")

# input image
st.subheader("Please Upload a Raw Image")

# file uploader
image = st.file_uploader("Please Upload a Raw Image", type = ["jpg", "png", "jpeg"], accept_multiple_files = False, help = "Upload an Image")

if image:
    user_image = Image.open(image)
    # save the image to set the path
    user_image.save(IMAGE_NAME)
    # set the user image
    st.image(user_image, caption = "User Uploaded Image")

    #get the features
    with st.spinner("Processing......."):
        image_features = featurization(IMAGE_NAME, ConvNeXtXLarge_featurized_model)
        model_predict = classification_model.predict(image_features)
        result_label = CLASS_LABEL[model_predict[0]]
        st.success(f"Prediction: {result_label}")