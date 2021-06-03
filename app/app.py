from os import path, remove
from random import randint

import streamlit as st
from PIL import Image

from model import get_model, get_model_summary, model_prediction

st.set_page_config(page_title="Real Fake Face Classification",
                   page_icon=path.join('assets', 'icons', 'logo.png'))

with open(path.join('assets', 'styles.css')) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

_, page_banner_img, _ = st.beta_columns([3, 2, 3])
page_banner_img.image(path.join('assets', 'icons', 'robot_face.png'),
                      use_column_width=True)

st.title('Real Fake Face Image Classifier "DeepFakes"')
st.subheader('')

st.markdown(
    """
    This is a real-fake face classifier built with convolutional neural networks. The classifier was trained on a
    data set comprised of 1400 images (700 of each class) and tested on 600 images (300 per class). The classifier
    achieved an accuracy of **83.2%**. You can find more performance metrics and information about this project in
    the [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/dvpinho/RealFakeFaces)
    repository. To use this web application just drag and drop a face image to be classified by the model. While
    you think about that, have a 🍪 and refresh the page once or twice to classify a few built-in faces embedded
    into the app. The classifier will return the result with the associated probability that a specific face image
    belongs to either the ```Real``` or ```Fake``` class. The model's architecture summary is also presented below.
    """
)

model_cnn = get_model(path.join('assets', 'model', 'model.hdf5'))
model_summary = get_model_summary(model_cnn)

# Print model summary to expandable container
with st.beta_expander(label='Model architecture summary', expanded=False):
    st.markdown(f"""```{model_summary}""")

# Upload face image file to use for prediction
uploaded_file = st.file_uploader('Upload a face image for prediction',
                                 type=['png', 'jpg', 'jpeg'],
                                 accept_multiple_files=False,
                                 help='Please select an image for prediction')

if uploaded_file is not None and uploaded_file.type.split('/')[0] == 'image':

    uploaded_file_name = uploaded_file.name
    uploaded_image = Image.open(uploaded_file).save(uploaded_file_name)
    st.image(Image.open(uploaded_file_name).resize((720, 720)))

    prediction_label, real_face_prob, fake_face_prob = model_prediction(uploaded_file_name, model_cnn)
    remove(uploaded_file_name)  # scrap the file after prediction

else:

    embedded_img = path.join('assets', 'faces', f'{randint(1, 6)}.jpg')
    st.image(Image.open(embedded_img).resize((720, 720)))
    prediction_label, real_face_prob, fake_face_prob = model_prediction(embedded_img, model_cnn)

with st.beta_expander(label='Prediction Probabilities', expanded=False):
    st.markdown('<h3>Results</h3>', unsafe_allow_html=True)

    row_11, row_12 = st.beta_columns([1, 1])
    row_11.markdown('<h4>Face Type</h4>', unsafe_allow_html=True)
    row_12.markdown('<h5>Probability</h5>', unsafe_allow_html=True)
    st.progress(0)

    row_21, row_22 = st.beta_columns([1, 1])
    row_21.markdown('<h3>REAL</h3>', unsafe_allow_html=True)
    row_22.markdown(f'<h2>{real_face_prob} %</h2>', unsafe_allow_html=True)
    st.progress(real_face_prob)

    row_31, row_32 = st.beta_columns([1, 1])
    row_31.markdown('<h3>FAKE</h3>', unsafe_allow_html=True)
    row_32.markdown(f'<h2>{fake_face_prob} %</h2>', unsafe_allow_html=True)
    st.progress(fake_face_prob)

st.markdown(f"The classifier's prediction is that the loaded image is a ```{prediction_label}```❗")
