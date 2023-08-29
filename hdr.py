import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
model=keras.models.load_model("Trained_model_cnn.h5")
st.title("Digit Recognizer")
size=256
canvas_result=st_canvas(fill_color="#ffffff",stroke_width=10,stroke_color='#ffffff',background_color="#000000",height=150,width=150,drawing_mode='freedraw',key="canvas")
if canvas_result.image_data is not None:
    img=cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
    img_rescale=cv2.resize(img,(size,size),interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescale)
if st.button('Predict'):
    Xt=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pred=model.predict(Xt.reshape(-1,28,28,1))
    st.write(f'result:{np.argmax(pred[0])}')
    st.bar_chart(pred[0])
