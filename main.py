import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import scipy.io
# import pathlib
# from tensorflow import keras
from src.visualization import plot_ecg

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title='🫀 ECG Classification',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide'
)

# PAge Intro
st.write("""
# 🫀 ECG Classification
For this app, we trained a model to detect heart anomalies based on the [Physionet 2017 Cardiology Challenge](https://physionet.org/content/challenge-2017/1.0.0/) 
dataset.
**Possible Predictions:** Normal, Myocardial infarction, Abnormal Heartbeat, History of MI
### Authors:
- Aarushi Dhanuka
- Sharanya Prabhu
- Shourya Gupta
- Gautham Prabhu

-------
""".strip())

#---------------------------------#
# Data preprocessing and Model building

@st.cache(allow_output_mutation=True)
def read_ecg_preprocessing(uploaded_ecg):

      FS = 300
      maxlen = 30*FS

      uploaded_ecg.seek(0)
      mat = scipy.io.loadmat(uploaded_ecg)
      mat = mat["val"][0]

      uploaded_ecg = np.array([mat])

      X = np.zeros((1,maxlen))
      uploaded_ecg = np.nan_to_num(uploaded_ecg) # removing NaNs and Infs
      uploaded_ecg = uploaded_ecg[0,0:maxlen]
      uploaded_ecg = uploaded_ecg - np.mean(uploaded_ecg)
      uploaded_ecg = uploaded_ecg/np.std(uploaded_ecg)
      X[0,:len(uploaded_ecg)] = uploaded_ecg.T # padding sequence
      uploaded_ecg = X
      uploaded_ecg = np.expand_dims(uploaded_ecg, axis=2)
      return uploaded_ecg

model_path = 'models/weights-best.hdf5'
classes = ['Normal','Atrial Fibrillation','Other','Noise']




# Visualization --------------------------------------
@st.cache(allow_output_mutation=True,show_spinner=False)
def visualize_ecg(ecg,FS):
    fig = plot_ecg(uploaded_ecg=ecg, FS=FS)
    return fig


#Formatting ---------------------------------#

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {	
            visibility: hidden;
        }
        footer:after {
            content:'Made for Machine Learning in Healthcare with Streamlit';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.sidebar.markdown("")

file_gts = {
    "A00001" : "Normal",
    "A00002" : "Normal",
    "A00003" : "Normal",
    "A00004" : "Atrial Fibrilation",
    "A00005" : "Other",
    "A00006" : "Normal",
    "A00007" : "Normal",
    "A00008" : "Other",
    "A00009" : "Atrial Fibrilation",
    "A00010" : "Normal",
    "A00015" : "Atrial Fibrilation",
    "A00205" : "Noise",
    "A00022" : "Noise",
    "A00034" : "Noise",
}
valfiles = [
    'None',
    'A00001.mat','A00010.mat','A00002.mat','A00003.mat',
    "A00022.mat", "A00034.mat",'A00009.mat',"A00015.mat",
    'A00008.mat','A00006.mat','A00007.mat','A00004.mat',
    "A00205.mat",'A00005.mat'
]

with st.sidebar.header('Use a file from the validation set'):
        pre_trained_ecg = st.sidebar.selectbox(
            'Select a file from the validation set',
            valfiles,
            format_func=lambda x: f'{x} ({(file_gts.get(x.replace(".mat","")))})' if ".mat" in x else x,
            index=1,

        )
        
        st.sidebar.markdown("ECG Images dataset of Cardiac Patients")

st.sidebar.markdown("---------------")
st.sidebar.markdown("Check the [Github Repository](https://github.com/simonsanvil/ECG-classification-MLH) of this project")
#---------------------------------#
# Main panel



    

    # st.line_chart(np.concatenate(ecg).ravel().tolist())