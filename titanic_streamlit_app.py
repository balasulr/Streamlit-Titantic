import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explore Titantic dataset with Streamlit", layout="wide")

st.title("Explore Titanic Dataset with Streamlit")
st.subheader("This Streamlit app explores the Titanic dataset to uncover survival insights and visualize passenger data")

# Load Titanic dataset from URL
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(data_url)
st.write(df)

