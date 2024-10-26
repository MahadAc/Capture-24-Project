import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
import features
import joblib
st.file_uploader("accelerometer data", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible") 
loaded_othermodel = joblib.load("/content/drive/MyDrive/Veritas - Mahad/random_forrest_model.joblib")
model = lambda x: 42

# Create a function to extract data for any participant
def extract_data(participant_number):
    #Load another participant data
    df = pd.read_csv(data_directory + f"/P{participant_number:03d}.csv.gz",
    index_col='time', parse_dates=['time'],
    dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
    # Simplify annotations
    df['label'] = (anno_label_dict['label:Willetts2018']
    .reindex(df['annotation'])
    .to_numpy())
    # Extract windows
    X_values, Y_values = extract_windows(df)
    # Extract features
    X_features = pd.DataFrame([features.extract_features(x) for x in X_values])
    return X_features, Y_values



def main(): 
    st.title("Accelerometer Data")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Accelerometer Data App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    if st.button("Predict"): 
        features = []
        features.extract_features
        
        # 4. This is where the model actually makes its predictions
        # You will need to change this to model.predict, once you've setup the 
        prediction = model(features)
        model.predict 
        # 5. Here is where we create an output to display to the user
        # Currently, it just returns some text. 
        # Instead, we want to create the figures showing participant activity over time (Similar to plot_compare() in the notebook)
        # I don't actually know how to display images in streamlit yet, you will need to look this up
        output = int(prediction)
        if output == 1:
            text = ">50K"
        else:
            text = "<=50K"
        
        st.success('Employee Income is {}'.format(text))


      
if __name__=='__main__': 
    main()
# Found this for image in streamlit
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


