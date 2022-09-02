import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from easyocr import Reader
import os
import numpy as np

@st.cache
def load_model(): 
    reader = Reader(['en', 'th'],model_storage_directory='.', gpu=False)
    return reader 

st.title("Recognize Text and Bounding Boxes")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def read_text(image_path, model_name, in_line = "True"):
    text = model_name.readtext(image_path, detail = 0, paragraph = in_line)
    return '\n'.join(text)

def draw_boxes(image, bounds, color='red', width=5):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


reader = load_model()


data = st.file_uploader("Upload your image", type=['.jpg', '.png', '.jpeg'])
st.set_option('deprecation.showfileUploaderEncoding', False)

try:
    
    if data is not None:
        image = Image.open(data)
        st.header("Your Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Drawing bounding boxes...'):
        bounds = reader.readtext(np.array(image), detail=1)
        img_new = draw_boxes(image, bounds)
        # img_new.save(f'TempDir\{data.name}')
    
    st.header("Image with Bounding Box")
    img1 = np.array(img_new)
    img2 = Image.fromarray(img1)
    st.image(img2, use_column_width=True)
    
    with st.spinner('Extracting text...'):
        op_text = read_text(np.array(image), reader)
        
    st.header("Text")
    st.write(op_text)
    
except:
    pass





