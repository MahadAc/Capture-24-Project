import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
import features
st.file_uploader("accelerometer data", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible") 






def main(): 
    st.title("Accelerometer Data")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Accelerometer Data App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

 if st.button("Predict"): 

# 2. Here is where you need to add code to convert the raw xyz data into features
# hint: use the method features.extract_features. To do this you will need to upload features.py (its on our shared drive) to your github
features = []
features.extract_features
