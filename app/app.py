"""App"""
import os
import glob

import streamlit as st
from PIL import Image

from ..executor import inferrer
#.inferrer
# import Inferrer

def file_selector(folder_path='./configs'):
    filenames = glob.glob(os.path.join(folder_path, '*.yml'))
    selected_filepath = st.sidebar.selectbox('Select config file', filenames)
    return selected_filepath

def image_selector():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    return st.sidebar.file_uploader(' ', type=['png', 'jpg', 'jpeg'])

st.title("Image Semantic Segmentation App")
st.write('\n')

# show main image
image = Image.open('./app/images/cat.png')
show = st.image(image, use_column_width=True)

# sidebar
st.sidebar.title('Upload Config and Image')
selected_path = file_selector()
uploaded_file = image_selector()
u_img = None

if selected_path is not None:
    inferrer = Inferrer(selected_path)

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)

st.sidebar.write('\n')

# Classify
if st.sidebar.button('Click Here to Classify'):
    if selected_path is None:
        st.sidebar.write('Please upload a Config file')
    elif uploaded_file is None:
        st.sidebar.write('Please upload an Image to Classify')
    else:
        with st.spinner('Classifying..'):
            results = inferrer.infer(u_img)
            st.success('Done!')

        st.sidebar.header('Predicts: ')
        probability = '{:.3f}'.format(results['prob'] * 100)

        class_name = inferrer.classes[results['label']]
        st.sidebar.write(f"Model: {inferrer.model_name.upper()}\n")
        st.sidebar.write(f"Class: '{class_name.capitalize()}'\n")
        st.sidebar.write(f"Probability: {probability}%")