import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Explore Titantic dataset with Streamlit", layout="wide")

# Title and subtitle
st.title("Explore Titanic Dataset with Streamlit")
st.subheader("This Streamlit app explores the Titanic dataset to uncover survival insights and visualize passenger data")

# Load Titanic dataset from URL
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(data_url)

# Displays the full dataframe
st.write(df)

# Dataset overview
st.subheader("Dataset Overview: Titanic Passenger Data")
st.subheader("The above dataset contains information about Titanic passengers, including whether they survived, their age, class, fare, and more")

# Column overview
st.subheader("Column Overview")

# Show column names
st.write("The columns in the dataset are: {}".format(", ".join(df.columns)))

# Column Descriptions
st.subheader("Column Descriptions")

# Create a dictionary for column descriptions
column_info = {
    "PassengerId": "Unique identifier for each passenger",
    "Survived": "Survival indicator (0 = No, 1 = Yes)",
    "Pclass": "Ticket class (1 = Upper, 2 = Middle, 3 = Lower)",
    "Name": "Full name, includes titles",
    "Sex": "Gender of the passenger",
    "Age": "Age in years (may contain missing values)",
    "SibSp": "Number of siblings/spouses aboard",
    "Parch": "Number of parents/children aboard",
    "Ticket": "Ticket number",
    "Fare": "Price paid for the ticket",
    "Cabin": "Cabin number (many missing values)",
    "Embarked": "Port of embarkation (C, Q, S)"
}

# Convert dictionary to DataFrame for better display
desc_df = pd.DataFrame.from_dict(column_info, orient='index', columns=["Description"])

# Display the descriptions in a table
st.table(desc_df)

# Initial data exploration
st.subheader("Initial Data Exploration")

# Show first few rows
st.subheader("Here are the first few rows of the dataset:")
st.write(df.head())

# Show dataset shape
st.write("The dataset has {} rows and {} columns.".format(df.shape[0], df.shape[1]))

# Data types overview
st.subheader("Data Types Overview to help distinguish between numerical features, categorical identifiers, and the target label:")
st.write(df.dtypes.value_counts())
st.write("There are 2 numerical features (float64), 5 categorical features (object), and 1 target label (int64).")

# Summary statistics
st.write("Summary statistics for all columns transposed:")
st.write(df.describe(include='all').T)

# Data Visualization
st.subheader("Graphs to visualize data distributions and relationships:")

st.header("Data Visualization: Choose Chart Type and Column")

# Two dropdowns in columns
col1, col2 = st.columns(2)

with col1:
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Pie Chart", "Scatter Plot", "Histogram", "Bar Chart"]
    )

with col2:
    # Dynamically choose column based on chart type
    if chart_type in ["Pie Chart", "Bar Chart"]:
        category = st.selectbox("Select Column", df.columns)

    elif chart_type == "Histogram":
        category = st.selectbox(
            "Select Numeric Column",
            df.select_dtypes(include=["int", "float"]).columns
        )

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox(
            "X-axis",
            df.select_dtypes(include=["int", "float"]).columns
        )
        y_col = st.selectbox(
            "Y-axis",
            df.select_dtypes(include=["int", "float"]).columns
        )

# Render chart
st.subheader("Visualization")

if chart_type == "Pie Chart":
    fig, ax = plt.subplots(figsize=(3, 3))
    df[category].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax,
        startangle=90,
        textprops={"fontsize": 8}
    )
    ax.set_ylabel("")
    st.pyplot(fig)

elif chart_type == "Bar Chart":
    fig, ax = plt.subplots(figsize=(4, 3))
    df[category].value_counts().plot(kind="bar", ax=ax, color="teal")
    ax.set_xlabel(category)
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif chart_type == "Histogram":
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df[category].dropna(), bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel(category)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif chart_type == "Scatter Plot":
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(df[x_col], df[y_col], alpha=0.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

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

