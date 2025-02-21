import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(layout="wide")
st.title("Penguins Species Classifier")
st.write("This application predict the species of the penguin based on the specifications")
st.write("***")

st.sidebar.header("Input Paramters")
Bill_Length=st.sidebar.slider("Bill Length", 39.0, 60.0, 45.5)
Bill_Depth=st.sidebar.slider("Bill Depth",13.0, 22.0, 15.5)
Flipper_Length=st.sidebar.slider("Flipper Length", 170, 235, 180)
Body_Mass=st.sidebar.slider("Body Mass", 2700, 6300, 3500)


train_data=pd.read_csv("penguins_cleaned.csv")
x_train=train_data[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y_train=train_data["species"]




test_data=pd.DataFrame({
    "bill_length_mm":Bill_Length,
    "bill_depth_mm":Bill_Depth,
    "flipper_length_mm":Flipper_Length,
    "body_mass_g":Body_Mass
        }, index=[0])

st.header("Input Parameters")
st.dataframe(test_data)
st.write("***")

target_names=pd.DataFrame({
    "species":["Adelie", "ChinStrap", "Gentoo"]
})
st.header("Species")
st.dataframe(target_names)
st.write("***")


model=RandomForestClassifier()
model=model.fit(x_train, y_train)
pred=model.predict(test_data)
st.header("prediction probability")
st.dataframe(model.predict_proba(test_data))
st.write("***")
st.header("prediction")
st.dataframe(pred)
