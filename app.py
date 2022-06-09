#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(testInstance):
    WQ = pd.read_csv("water_potability.csv")
    
    # Impute missing values using KNNImputer
    Before_imputation = WQ
    imputer = KNNImputer(n_neighbors=4)
    After_Imputation = imputer.fit_transform(Before_imputation)
    WQI = pd.DataFrame(After_Imputation)
    WQI.rename(columns = {0:'ph', 1:'Hardness', 2:'Solids', 3:'Chloramines', 4:'Sulfate', 5:'Conductivity', 6:'Organic_carbon', 7:'Trihalomethanes', 8:'Turbidity', 9:'Potability'}, inplace = True)

    # RSeparate the data set columns in to dependant and independant variables
    X = WQI.drop('Potability',axis=1).values
    y = WQI['Potability'].values

    # Split the dataset into train test parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    
    # Standardise the testInstance data
    scaler = StandardScaler()
    scaler.fit(X_train)
    testingData= scaler.transform(testInstance)
    
    predict = classifier.predict(testingData)
    
    # Return Predicted result   
    if predict == 0.0:
        pred = 'NOT POTABLE'
    else:
        pred = 'POTABLE'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Water Quality ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    ph = st.number_input('ph')
    Hardness = st.number_input('Hardness')
    Solids = st.number_input('Solids)')
    Chloramines = st.number_input("Chloramines") 
    Sulfate = st.number_input("Sulfate")
    Conductivity = st.number_input("Conductivity")
    Organic_carbon = st.number_input('Organic_carbon')
    Trihalomethanes = st.number_input('Trihalomethanes')
    Turbidity = st.number_input('Turbidity)')
    
    
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        testInstance = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]
        result = prediction(testInstance)
        # result = prediction(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity) 
        st.success('Water {}'.format(result))
        
     
if __name__=='__main__': 
    main()

