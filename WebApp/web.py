import sys
sys.path.append('../')
# streamlit import
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import base64

# model part
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img, array_to_img, img_to_array

import matplotlib.pyplot as plt
from skimage.io import imshow
from Segmentation.image_segmentation import dice_coef, DiceLoss, IoU_coeff, IoULoss, intersection
from Image_Classification.model_cnn import processing_image

import pandas as pd
import numpy as np
from PIL import Image

#'''
#Option Menu, including:
#    - About: In introduction about the solution / Tutorial about how to use it
#    - Photo Analysis: 1. Upload Image; 2. show if there's a silo or not; 3. Show segmentation results; 4. Density Analysis
#    - Contact/Team: Team intro and contact form
#'''
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)



st.set_page_config(layout="wide")
CLF_MODEL_PATH = "test_.checkpoint"
MODEL_PATH = "model_v2"
THRESHOLD = 0.5
CLF_THRESHOLD = 0.5

clf_model = model_load = keras.models.load_model(CLF_MODEL_PATH)
seg_model = keras.models.load_model(MODEL_PATH, custom_objects={'intersection': intersection, 'dice_coef': dice_coef, 'DiceLoss':  DiceLoss, 'IoU_coeff': IoU_coeff, 'IoULoss': IoULoss})

with st.sidebar:
    choose = option_menu("Foodix", ["About", "Silo Detection", "Maps", "Team", "Contact"],
                            icons=['house', 'camera fill', 'person lines fill'],
                            menu_icon="app-indicator", default_index=0,
                            #orientation='horizontal',
                            styles={
            "container": {"padding": "5!important", "background-color": "#323232"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
        )


logo = Image.open(r'Foodix.png')
profile = Image.open(r'example.png')
if choose == "About":
    set_background('background.png')
    st.markdown(""" <style> .font1 {
        font-size:150px ; font-family: 'Cooper Black'; color: #000000;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<h1 class="font1">Foodix</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns( [0.5, 0.5])
    with col2:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FFFFFF;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">We help diminish global famine crisis using AI.</p>', unsafe_allow_html=True)    
#    with col2:               # To display brand log
#        st.image(logo, width=130 )
    
#    st.write("Sharone Li is a data science practitioner, enthusiast, and blogger. She writes data science articles and tutorials about Python, data visualization, Streamlit, etc. She is also an amateur violinist who loves classical music.\n\nTo read Sharone's data science posts, please visit her Medium blog at: https://medium.com/@insightsbees")    
#    st.image(profile, width=400 )

elif choose == "Silo Detection":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
        
    with col2:               # To display brand logo
        st.image(logo,  width=150)
    #Add file uploader to allow users to upload photos
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Original Image</p>',unsafe_allow_html=True)
            st.image(image,width=300)  

        with col2:
            st.markdown('<p style="text-align: center;">Image Segmentation</p>',unsafe_allow_html=True)
            img = load_img(uploaded_file, grayscale=False)  #target_size=(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT) 
            img = img_to_array(img)
            img = img.astype('float32') / 255.0
            imshow(img)
            plt.show()
            img = np.expand_dims(img, axis=0)
            predictions = seg_model.predict(img)

            #imshow(np.squeeze(predictions[0]))
            #plt.show()
            #imshow(np.squeeze(np.array(predictions[0]>THRESHOLD, dtype=np.uint8)))
            #plt.show()
            res = (predictions[0] * 255).astype('int')
            st.image(res, width=300)
        
        img = processing_image(uploaded_file) 
        img = img.reshape(1,224,224,3)
        prediction = np.round(clf_model.predict(img))[0][0]
        if prediction >= CLF_THRESHOLD:
            st.markdown('<h3 style="color:white;">Yes, there is a silo in this area!</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color:white;">No, there is not any silo in this area</h3>', unsafe_allow_html=True)

elif choose == "Maps":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">The Map</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    df = pd.read_excel("LatLon.xlsx")
    st.map(df, zoom=6)

elif choose == "Team":
    st.markdown('<h3 style="color:white;">Our Data Scientist Team</h3>', unsafe_allow_html=True)

elif choose == "Contact":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')


#'''
#st.markdown('<h1 style="color:black;">Vgg 19 Image classification model</h1>', unsafe_allow_html=True)
#st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
#st.markdown('<h3 style="color:gray;"> street,  buildings, forest, sea, mountain, glacier</h3>', unsafe_allow_html=True)



#upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
#c1, c2= st.columns(2)
#if upload is not None:
#  im= Image.open(upload)
#  img= np.asarray(im)
#  image= cv2.resize(img,(224, 224))
#  #img= preprocess_input(image)
#  img= np.expand_dims(img, 0)
#  c1.header('Input Image')
#  c1.image(im)
#  c1.write(img.shape)
#'''


