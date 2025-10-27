import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explore Titantic dataset with Streamlit", layout="wide")

st.title("Explore Titanic Dataset with Streamlit")
st.subheader("This Streamlit app explores the Titanic dataset to uncover survival insights and visualize passenger data")

# Load Titanic dataset from URL
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(data_url)
st.write(df)

# See if have missing values
st.subheader("Missing Values Overview")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing %": missing_percent
})
st.dataframe(missing_df[missing_df["Missing Count"] > 0])

st.write("There are missing values in the dataset")

