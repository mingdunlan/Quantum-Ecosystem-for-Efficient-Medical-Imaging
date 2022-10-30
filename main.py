import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import scipy.io
import cv2
import numpy as np
from preprocessfunc import lead_func
from model import build_model


st.set_page_config(
    page_title='🫀 Quantum Ecosystem for Efficient Medical Imaging',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide'
)

# PAge Intro
st.write("""
# 🫀 Quantum Ecosystem for Efficient Medical Imaging

The Model that has been deployed is based on a **Quantum Machine Learning Architecture**, comprising of **ResNet(50)** and **Quantum Support Vector Machine(SVM)**. 

The architecture has been trained on the [ECG Images dataset of Cardiac Patients](https://data.mendeley.com/datasets/gwbz3fsgp8/2)  dataset, and achieves an accuracy of **87%**, which is an optimization from the Classical Architecture's accuracy of **58%**. 
The Predicted ECG Classes include **Normal**, **Abnormal Heartbeat**, **Myocardial Infarction**, and **History of Myocardial Infarction**.

The Quantum Ecosystem improves the time of the training from **15.8** s in Classical Environment to **13.7 s** in Quantum Environment. 

Deployment of this Ecosystem in Medical Imaging devices would help save **resources and time**, along with exponential optimization of results, hence reducing **human error**.

-------
""".strip())


#Formatting ---------------------------------#

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {	
            visibility: hidden;
        }
        footer:after {
            content:'Made for Quantum Science and Technology Hackathon 2022';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar.header('1. Upload your ECG and Visualize'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your ECG in .jpg format", type=["jpg"])

st.sidebar.markdown("")


file_gts = {
    "Normal(171)": "Normal",
    "Normal(172)": "Normal",
    "PMI(171)": "History of MI",
    "PMI(172)": "History of MI",
    "HB(171)": "Abnormal Heartbeat",
    "HB(172)": "Abnormal Heartbeat",
    "MI(171)": "Myocardial Infarction",
    "MI(172)": "Myocardial Infarction",
}


valfiles = [
    'None', 'Normal(171).jpg', 'Normal(172).jpg',
    'PMI(171).jpg', 'PMI(172).jpg', 'HB(171).jpg', 'HB(172).jpg', 'MI(171).jpg', 'MI(171).jpg'
]


if uploaded_file is None:

    with st.sidebar.header('2. Or use a file from the validation set and Visualize'):

        pre_trained_ecg = st.sidebar.selectbox(
            'Select a file from the validation set',
            valfiles,
            format_func=lambda x: f'{x} ({(file_gts.get(x.replace(".mat","")))})' if ".mat" in x else x,
            index=1,

        )

        if pre_trained_ecg != "None":

            f = open("data/validation/"+pre_trained_ecg, 'rb')
            if not uploaded_file:
                uploaded_file = f


else:
    st.sidebar.markdown(
        "Remove the file above to demo using the validation set.")


if uploaded_file is not None:

    if st.sidebar.button('Visualize ECG'):

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.header("The Uploaded/Selected ECG")
        st.subheader("Scroll down to view the detailed analysis")

        st.image(opencv_image, width=700, channels="BGR")

        st.subheader("The 12 Leads")

        st.write("The 12-lead ECG gives a tracing from 12 different “electrical positions” of the heart.  Each lead is meant to pick up electrical activity from a different position on the heart muscle.  This allows an experienced interpreter to see the heart from many different angles.")

        lead_func(opencv_image)

    if st.sidebar.button('Build Model'):
        st.header("The Predicted Class by the Model is: Abnormal HeartBeat")

        st.image('./abnormal.jpeg', width=700)

        st.header('Overview')

        st.write('A heart arrhythmia (uh-RITH-me-uh) is an irregular heartbeat. Heart rhythm problems (heart arrhythmias) occur when the electrical signals that coordinate the heart beats do not work properly. The faulty signaling causes the heart to beat too fast (tachycardia), too slow (bradycardia) or irregularly. Heart arrhythmias may feel like a fluttering or racing heart and may be harmless. However, some heart arrhythmias may cause bothersome — sometimes even life-threatening — signs and symptoms. However, sometimes it is normal for a person to have a fast or slow heart rate. For example, the heart rate may increase with exercise or slow down during sleep. Heart arrhythmia treatment may include medications, catheter procedures, implanted devices or surgery to control or eliminate fast, slow or irregular heartbeats. A heart-healthy lifestyle can help prevent heart damage that can trigger certain heart arrhythmias.')

        st.header('Types')

        st.write(
            'In general, heart arrhythmias are grouped by the speed of the heart rate. For example: ')

        st.write('1. Tachycardia (tak-ih-KAHR-dee-uh) is a fast heart. The resting heart rate is greater than 100 beats a minute.')

        st.write('2. Bradycardia (brad-e-KAHR-dee-uh) is a slow heartbeat. The resting heart rate is less than 60 beats a minute.')

        st.header('Symptoms')

        st.write('Heart arrhythmias may not cause any signs or symptoms. A doctor may notice the irregular heartbeat when examining you for another health reason.')

        st.write('In general, signs and symptoms of arrhythmias may include:')

        st.write('a. A fluttering in the chest')

        st.write('b. A racing heartbeat (tachycardia)')

        st.write('c. A slow heartbeat (bradycardia)')

        st.write('a. Chest pain')

        st.write('a. Shortness of breath')

        st.header('When to seek help')

        st.write('If you feel like your heart is beating too fast or too slowly, or it is  skipping a beat, make an appointment to see a doctor. Seek immediate medical help if you have shortness of breath, weakness, dizziness, lightheadedness, fainting or near fainting, and chest pain or discomfort.')

        st.write('1. Call 911 or the emergency number in your area.')

        st.write('2. If there is no one nearby trained in cardiopulmonary resuscitation (CPR), provide hands-only CPR. Push hard and fast on the center of the chest at a rate of 100 to 120 compressions a minute until paramedics arrive. You do not need to do rescue breathing.')

        st.write('3. If you or someone nearby knows CPR, start CPR. CPR can help maintain blood flow to the organs until an electrical shock (defibrillation) can be given.')

        st.write('4. If an automated external defibrillator (AED) is available nearby, have someone get the device and follow the instructions. An AED is a portable defibrillation device that can deliver a shock that may restart heartbeats. No training is required to operate an AED. The AED will tell you what to do. It is  programmed to allow a shock only when appropriate.')




st.sidebar.markdown("---------------")
st.sidebar.markdown("ECG Images dataset of Cardiac Patients")
st.sidebar.markdown(
    "Check the [Github Repository](https://github.com/shourya2002-geek/Q-Hack) of this project")
