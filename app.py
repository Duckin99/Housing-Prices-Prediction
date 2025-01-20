import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("Housing Prices Competition for Kaggle Learn Users")
st.write("""
Using *Random Forest* model for **Housing Prices** Prediction
""")

df = pd.read_csv("train.csv", index_col="Id")

st.write(f"Raw data: {df.shape[0]} rows x {df.shape[1]} columns")
st.dataframe(df)

df = df.dropna()

st.code(df.describe())

X = df