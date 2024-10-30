import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
import features
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

model = joblib.load("random_forrest_model.joblib")

#Create function to extract windows
def make_windows(data, winsec=10, sample_rate=100, dropna=True, verbose=False, label_col='label'):
    X, Y, T = [], [], []

    if label_col not in data.columns:
        return_y = False
    else:
        return_y = True

    for t, w in tqdm(data.resample(f"{winsec}s", origin='start'), disable=not verbose):

        if len(w) < 1:
            continue

        t = t.to_numpy()

        x = w[['x', 'y', 'z']].to_numpy()

        if return_y:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unable to sort modes")
                y = w[label_col].mode(dropna=False).iloc[0]
    
            if dropna and pd.isna(y):  # skip if annotation is NA
                continue

        if not is_good_window(x, sample_rate, winsec):  # skip if bad window
            continue

        X.append(x)
        if return_y:
            Y.append(y)
        T.append(t)

    X = np.stack(X)
    T = np.stack(T)
    if return_y:
        Y = np.stack(Y)
    else:
        Y = None
    return X, Y, T

def is_good_window(x, sample_rate, winsec):
    ''' Check there are no NaNs and len is good '''

    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) != window_len:
        return False

    # Check no nans
    if np.isnan(x).any():
        return False

    return True
            
# Create a function to extract data for any participant
def extract_data(df):
    #Load another participant data
    if df is None:
        df = pd.read_csv(
            "./example_data_P142.csv.gz",
            index_col='time', parse_dates=['time'],
            dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}
        )
        #Simplify annotations
        anno_label_dict = pd.read_csv('annotation-label-dictionary.csv', index_col = 'annotation', dtype = 'string')
        df['label'] = (anno_label_dict['label:Willetts2018'].reindex(df['annotation']).to_numpy())
    # Extract windows
    X, Y, T = make_windows(df, winsec=10, label_col="label")
    # Extract features
    X_features = pd.DataFrame([features.extract_features(x) for x in X])
    return X_features, Y, T

def plot_figure(prediction, T, Y=None):
    fig, ax = pt.subplots()
    ax.plot(np.arange(10), np.arange(10)**2)
    return fig, ax

def main(): 
    st.title("Accelerometer Data")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Accelerometer Data App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    df_upload = st.file_uploader("accelerometer data", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible") 

    if st.button("Predict"): 
        # 4. This is where the model actually makes its predictions
        # You will need to change this to model.predict, once you've setup the 
        if df_upload is not None:
            df = pd.read_csv(df_upload, index_col='time', parse_dates=['time'], dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}, compression="gzip")
        else:
            df = None
        with st.spinner('Extracting features...'):
            X_features, Y, T = extract_data(df)
            
        with st.spinner('Running model...'):
            prediction = model.predict(X_features) 

        with st.spinner('Creating figure...'):
            fig = plot_figure(prediction, T, Y)
        st.pyplot(fig=fig)


      
if __name__=='__main__': 
    main()


