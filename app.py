import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler

st.title("Housing Prices Competition for Kaggle Learn Users")
st.write("""
Using *Random Forest* model for **Housing Prices** Prediction
""")

@st.cache_data
def load_data(name = "test.csv"):
    df = pd.read_csv(name, index_col=0)
    return df

X_test = load_data()

@st.cache_data
def load_model(name = "model.pkl"):
    _model = joblib.load(name)
    return _model

model = load_model()
cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']

st.code(f"test.csv\nDimension: {X_test.shape[0]} rows x {X_test.shape[1]} columns")
st.dataframe(X_test)

@st.cache_data
def show_shap(_model, X_test):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap.png")

st.write("### Select Features")
inputs = st.multiselect("Input Features", cols)

df_train = load_data("train.csv")

predictions = model.predict(df_train[cols])

def create_graph(inputs):
    fig, ax = plt.subplots()
    for input in inputs:
        X_train = df_train[input]
        y_train = df_train["SalePrice"]
        ax.scatter((X_train - X_train.mean())/X_train.std(), y_train)
    return fig

st.pyplot(create_graph(inputs))

st.write("---")
st.write("""
### SHAP Value
SHAP (SHapley Additive exPlanations) value is a metric used to explain the output of machine learning models by quantifying the contribution of each feature to the model's prediction.
""")
st.image("shap.png")

show_shap(model, X_test[cols])
