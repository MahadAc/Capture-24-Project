import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

model = lambda x: 42
cols=['age','workclass','education','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss',
      'hours-per-week','native-country']    
  
def main(): 
    st.title("Income Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
   st.file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible") 
    
    if st.button("Predict"): 
        features = [[age,workclass,education,marital_status,occupation,relationship,race,gender,capital_gain,capital_loss,hours_per_week,nativecountry]]
        data = {'age': int(age), 'workclass': workclass, 'education': education, 'maritalstatus': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capitalgain': int(capital_gain), 'capitalloss': int(capital_loss), 'hoursperweek': int(hours_per_week), 'nativecountry': nativecountry}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['age','workclass','education','maritalstatus','occupation','relationship','race','gender','capitalgain','capitalloss','hoursperweek','nativecountry'])
                
        category_col =['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
            
        features_list = df.values.tolist()      
        prediction = model(features_list)
    
        output = int(prediction)
        if output == 1:
            text = ">50K"
        else:
            text = "<=50K"

        st.success('Employee Income is {}'.format(text))
      
if __name__=='__main__': 
    main()
git add test.py
git commit -m "First update"
git push