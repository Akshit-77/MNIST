import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache_resource
def load_model():
  model = tf.keras.models.load_model('my_model.hdf5')
  return model
model = load_model()

st.write("Digit Classifier")

file = st.file_uploader("please upload image of a digit",type=['jpg','png'])



def import_and_predict(image_data,model):
  size=(28,28)
  #image = ImageOps.fit(image_data,size)
  image = image_data.resize(size,Image.LANCZOS)
  image = image.convert('L')
  img_array = np.array(image)/255
  img_reshape = img_array.reshape(1,28,28,1)
  prediction = model.predict(img_reshape)
  return prediction.argmax(axis = 1)

if file is None:
  st.text('please upload an image')
else:
  image = Image.open(file)
  st.image(image,use_column_width=True)
  predictions = import_and_predict(image,model)
  string = 'The image uploaded is digit ',str(predictions)
  st.success(string)
