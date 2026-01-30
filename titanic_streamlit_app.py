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
st.header("Data Visualization")

# Two dropdowns for selecting chart type and columns
# Create two columns for layout
col1, col2 = st.columns(2)

# Chart type selection
chart_type = col1.selectbox(
    "Select Chart Type",
    [ "Pie Chart",
     "Bar Chart",
     "Histogram",
     "Scatter Plot (Simple)",
     "Scatter Plot (Complex)",
     "Heatmap (Categorical)",
     "Heatmap (Numerical Correlation)"
    ]
)

# Dynamic column selection based on chart type
with col2:
    # Depending on chart type, show relevant dropdowns
    if chart_type in ["Pie Chart", "Histogram"]:
        category = st.selectbox("Select Column", df.columns, key="cat_select")
    
    # Additional selections for other chart types
    elif chart_type == "Bar Chart":
        bar_x = st.selectbox("X-axis Category", df.columns, key="bar_x")
        bar_hue = st.selectbox("Group By", df.columns, key="bar_hue")
    
    # Scatter Plot (Simple)
    elif chart_type == "Scatter Plot (Simple)":
        x_col = st.selectbox("X-axis", df.columns, key="simple_x")
        y_col = st.selectbox("Y-axis", df.columns, key="simple_y")
    
    # Scatter Plot (Complex)
    elif chart_type == "Scatter Plot (Complex)":
        x_col = st.selectbox("X-axis", df.columns, key="complex_x")
        y_col = st.selectbox("Y-axis", df.columns, key="complex_y")
        hue_col = st.selectbox("Color By", df.columns, key="complex_hue")
    
    # Heatmap (Categorical)
    elif chart_type == "Heatmap (Categorical)":
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols += ["Pclass", "Survived", "Embarked", "Sex"]

        heat_x = st.selectbox("Heatmap X-axis", categorical_cols, index=categorical_cols.index("Pclass"), key="cat_heat_x")
        heat_y = st.selectbox("Heatmap Y-axis", categorical_cols, index=categorical_cols.index("Survived"), key="cat_heat_y")
    
    # Heatmap (Numerical Correlation)
    elif chart_type == "Heatmap (Numerical Correlation)":
        st.write("Shows correlation between numerical features.")
        # No dropdowns needed
        pass

# Pie Chart
if chart_type == "Pie Chart":
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # Create pie chart
    df[category].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax,
        startangle=90,
        textprops={"fontsize": 8}
    )
    ax.set_ylabel("")
    ax.set_title(f"{category} Distribution", fontsize=12)
    st.pyplot(fig)

# Bar Chart
elif chart_type == "Bar Chart":
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x=bar_x, hue=bar_hue, ax=ax)
    ax.set_title(f"{bar_x} by {bar_hue}")
    st.pyplot(fig)

# Histogram
elif chart_type == "Histogram":
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Checks if the column is categorical or numerical
    if df[category].dtype == "object":
        # Categorical histogram
        df[category].value_counts().plot(kind="bar", ax=ax, color="purple")
        ax.set_title(f"{category} Frequency")
    else:
        # Numerical histogram
        ax.hist(df[category].dropna(), bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {category}")
    ax.set_xlabel(category)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Simple Scatter
elif chart_type == "Scatter Plot (Simple)":
    fig, ax = plt.subplots(figsize=(4, 3))

    # Convert categorical to numeric for scatter
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == "object":
            df_numeric[col] = df_numeric[col].astype("category").cat.codes
    ax.scatter(df_numeric[x_col], df_numeric[y_col], alpha=0.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    st.pyplot(fig)

# Complex Scatter
elif chart_type == "Scatter Plot (Complex)":
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    ax.set_title(f"{x_col} vs {y_col} colored by {hue_col}")
    st.pyplot(fig)

# Categorical Heatmap
elif chart_type == "Heatmap (Categorical)":    
    df_clean = df[[heat_x, heat_y]].dropna()
    heatmap_data = df_clean.pivot_table(index=heat_y, columns=heat_x, aggfunc="size", fill_value=0)
    
    # Warn if too many categories
    if heatmap_data.shape[0] > 50 or heatmap_data.shape[1] > 50:
        st.warning("This heatmap has too many categories and may take a long time to render.")

    # Check if heatmap_data is empty
    if heatmap_data.empty:
        st.write("No data available for the selected combination.")
    else:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt="d", ax=ax)
        ax.set_title(f"Heatmap: {heat_y} vs {heat_x}")
        st.pyplot(fig)

# Numerical Correlation Heatmap
elif chart_type == "Heatmap (Numerical Correlation)":
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Numerical Correlation Heatmap")
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
st.write("There are missing values present in the dataset")