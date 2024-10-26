import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
import features
import joblib
df_upload = st.file_uploader("accelerometer data", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible") 
model = joblib.load("random_forrest_model.joblib")

#Create function to extract windows
def extract_windows(data, winsize='10s'):
        X, Y = [], []
        for t, w in data.resample(winsize, origin='start'):
            # Check window has no NaNs and is of correct length
            # 10s @ 100Hz = 1000 ticks
            if w.isna().any().any() or len(w) != 1000:
                continue

            x = w[['x', 'y', 'z']].to_numpy()
            y = w['label'].mode(dropna=False).item()

            X.append(x)
            Y.append(y)

        X = np.stack(X)
        Y = np.stack(Y)
        return X, Y

# Create a function to extract data for any participant
def extract_data(df):
    #Load another participant data
    #df = pd.read_csv(f"/P{participant_number:03d}.csv.gz",
    #index_col='time', parse_dates=['time'],
    #dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
    #Simplify annotations
    anno_label_dict = pd.read_csv('annotation-label-dictionary.csv', index_col = 'annotation', dtype = 'string')
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
        # 4. This is where the model actually makes its predictions
        # You will need to change this to model.predict, once you've setup the 
        X_features, Y_values = extract_data(df_upload)
        prediction = model.predict(X_features) 
        # 5. Here is where we create an output to display to the user
        # Currently, it just returns some text. 
        # Instead, we want to create the figures showing participant activity over time (Similar to plot_compare() in the notebook)
        # I don't actually know how to display images in streamlit yet, you will need to look this up
        st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


      
if __name__=='__main__': 
    main()


