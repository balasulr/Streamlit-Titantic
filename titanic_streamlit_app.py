import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Explore Titantic dataset with Streamlit", layout="wide")

st.title("Explore Titanic Dataset with Streamlit")
st.subheader("This Streamlit app explores the Titanic dataset to uncover survival insights and visualize passenger data")

# Load Titanic dataset from URL
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(data_url)

# Displays the full dataframe
st.write(df)

st.subheader("The above dataset contains information about Titanic passengers, including whether they survived, their age, class, fare, and more")

# Column overview
st.subheader("Column Overview")

# Show column names
st.write("The columns in the dataset are: {}".format(", ".join(df.columns)))

# Column Descriptions
st.subheader("Column Descriptions")

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

desc_df = pd.DataFrame.from_dict(column_info, orient='index', columns=["Description"])
st.table(desc_df)

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

# Graphs
st.subheader("Graphs to visualize data distributions and relationships:")

# Gender Distribution Pie Chart
st.subheader("Gender Distribution Pie Chart:")
st.write("The pie chart below shows the distribution of male and female passengers on the Titanic.")
# Create two columns for layout
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Gender Distribution")
    # Create compact pie chart
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    # Plot pie chart
    df["Sex"].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 8}
    )
    # Clean up the pie chart
    ax.set_ylabel("")
    ax.set_title("Gender Distribution", fontsize=10)
    # Display the pie chart in Streamlit
    st.pyplot(fig)
with col2:
    st.markdown("### Summary:")
    st.write("This chart shows the proportion of male and female passengers in the Titanic dataset.")
    st.write("You can compare this with survival rates by gender to explore deeper insights.")


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

