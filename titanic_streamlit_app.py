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

    # Provide insights
    st.subheader("Insights / Analysis")
    top_value = df[category].value_counts().idxmax()
    top_count = df[category].value_counts().max()

    st.write(f"""
    • The most common category in **{category}** is **{top_value}** with **{top_count} passengers**.  
    • This distribution helps identify dominant groups in the dataset.  
    """)

# Bar Chart
elif chart_type == "Bar Chart":
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x=bar_x, hue=bar_hue, ax=ax)
    ax.set_title(f"{bar_x} by {bar_hue}")
    st.pyplot(fig)

    # Provide insights
    st.subheader("Insights / Analysis")
    largest_group = df.groupby([bar_x, bar_hue]).size().idxmax()
    largest_count = df.groupby([bar_x, bar_hue]).size().max()

    st.write(f"""
    • The largest group is **{largest_group}** with **{largest_count} passengers**.  
    • This chart highlights how **{bar_hue}** varies across **{bar_x}**.  
    """)

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

    # Provide insights
    st.subheader("Insights / Analysis")

    if df[category].dtype == "object":
        most_common = df[category].value_counts().idxmax()
        st.write(f"""
        • The most frequent category in **{category}** is **{most_common}**.  
        • This helps identify dominant categorical groups.  
        """)
    else:
        mean_val = df[category].mean()
        st.write(f"""
        • The average value of **{category}** is **{mean_val:.2f}**.  
        • The histogram shows how values are distributed around this mean.  
        """)

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

    # Provide insights
    st.subheader("Insights / Analysis")
    corr_val = df_numeric[[x_col, y_col]].corr().iloc[0,1]

    st.write(f"""
    • The correlation between **{x_col}** and **{y_col}** is **{corr_val:.2f}**.  
    • This indicates the strength and direction of their relationship.  
    """)


# Complex Scatter
elif chart_type == "Scatter Plot (Complex)":
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    ax.set_title(f"{x_col} vs {y_col} colored by {hue_col}")
    st.pyplot(fig)

    # Provide insights
    st.subheader("Insights / Analysis")
    st.write(f"""
    • This chart shows how **{y_col}** varies with **{x_col}** across different **{hue_col}** groups.  
    • Look for clustering or separation between colors to identify group differences.  
    """)


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

        # Provide insights
        st.subheader("Insights / Analysis")
        max_cell = heatmap_data.values.max()
        max_loc = list(zip(*divmod(heatmap_data.values.argmax(), heatmap_data.shape)))

        st.write(f"""
        • The highest count is **{max_cell}**, indicating the strongest category combination.  
        • Darker cells represent larger passenger groups.  
        • This heatmap highlights how **{heat_x}** and **{heat_y}** interact.  
        """)

# Numerical Correlation Heatmap
elif chart_type == "Heatmap (Numerical Correlation)":
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Numerical Correlation Heatmap")
    st.pyplot(fig)

    # Provide insights
    st.subheader("Insights / Analysis")
    surv_corr = corr['Survived'].sort_values(ascending=False)

    st.write(f"""
    • The strongest positive correlation with survival is **{surv_corr.index[1]} ({surv_corr.iloc[1]:.2f})**.  
    • The strongest negative correlation with survival is **{surv_corr.index[-1]} ({surv_corr.iloc[-1]:.2f})**.  
    • This helps identify which numeric features matter most.  
    """)

# Data Cleaning
st.header("Data Cleaning")

df_cleaned = df.copy()

# Handle missing values
df_cleaned['Age'].fillna(df['Age'].median(), inplace=True)
df_cleaned['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column since it has too many missing values
df_cleaned.drop(columns=['Cabin'], inplace=True)

# Fix data types
df_cleaned['Pclass'] = df_cleaned['Pclass'].astype('category')
df_cleaned['Survived'] = df_cleaned['Survived'].astype('category')

# Feature engineering
df_cleaned['FamilySize'] = df_cleaned['SibSp'] + df_cleaned['Parch'] + 1
df_cleaned['IsAlone'] = (df_cleaned['FamilySize'] == 1).astype(int)

# Drop original columns (SibSp, Parch) used to create FamilySize
df_cleaned.drop(columns=['SibSp', 'Parch'], inplace=True)

# Display cleaned data preview
st.write("Preview of cleaned data:")
st.write(df_cleaned.head())

st.subheader("Missing Values Overview (After Cleaning)")
missing_counts_clean = df_cleaned.isnull().sum()
missing_percent_clean = (missing_counts_clean / len(df_cleaned)) * 100
missing_df_clean = pd.DataFrame({
    "Missing Count": missing_counts_clean,
    "Missing %": missing_percent_clean
})
st.dataframe(missing_df_clean[missing_df_clean["Missing Count"] > 0])

# Display the full cleaned dataframe
st.subheader("Full Cleaned Dataset:")
st.write(df_cleaned)

# Post-cleaning EDA & Visualizations
st.header("4. Visualizations & Insights (Using Cleaned Data)")

st.subheader("Graphs to visualize data distributions and relationships:")
st.caption("All visualizations below use the cleaned dataset.")

col1, col2 = st.columns(2)

chart_type = col1.selectbox(
    "Select Chart Type",
    [
        "Pie Chart",
        "Bar Chart",
        "Histogram",
        "Scatter Plot (Simple)",
        "Scatter Plot (Complex)",
        "Heatmap (Categorical – Counts)",
        "Heatmap (Categorical – Percentages)",
        "Heatmap (Numerical Correlation)"
    ]
)

with col2:
    if chart_type in ["Pie Chart", "Histogram"]:
        category = st.selectbox("Select Column", df_cleaned.columns, key="cat_select_clean")
    
    elif chart_type == "Bar Chart":
        bar_x = st.selectbox("X-axis Category", df_cleaned.columns, key="bar_x")
        bar_hue = st.selectbox("Group By", df_cleaned.columns, key="bar_hue")
    
    elif chart_type == "Scatter Plot (Simple)":
        x_col = st.selectbox("X-axis", df_cleaned.columns, key="simple_x")
        y_col = st.selectbox("Y-axis", df_cleaned.columns, key="simple_y")
    
    elif chart_type == "Scatter Plot (Complex)":
        x_col = st.selectbox("X-axis", df_cleaned.columns, key="complex_x")
        y_col = st.selectbox("Y-axis", df_cleaned.columns, key="complex_y")
        hue_col = st.selectbox("Color By", df_cleaned.columns, key="complex_hue")
    
    elif chart_type in ["Heatmap (Categorical – Counts)", "Heatmap (Categorical – Percentages)"]:
        categorical_cols = df_cleaned.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols += ["Pclass", "Survived", "Embarked", "Sex"]
        categorical_cols = list(dict.fromkeys([c for c in categorical_cols if c in df_cleaned.columns]))
        
        heat_x = st.selectbox("Heatmap X-axis", categorical_cols, index=categorical_cols.index("Pclass"), key="cat_heat_x")
        heat_y = st.selectbox("Heatmap Y-axis", categorical_cols, index=categorical_cols.index("Survived"), key="cat_heat_y")
    
    elif chart_type == "Heatmap (Numerical Correlation)":
        st.write("Shows correlations between numerical features.")
        pass

# Pie chart
if chart_type == "Pie Chart":
    fig, ax = plt.subplots(figsize=(3, 3))
    df_cleaned[category].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax,
        startangle=90,
        textprops={"fontsize": 8}
    )
    ax.set_ylabel("")
    ax.set_title(f"{category} Distribution", fontsize=12)
    st.pyplot(fig)
        
    st.subheader("Insights / Analysis")
    top_value = df_cleaned[category].value_counts().idxmax()
    top_count = df_cleaned[category].value_counts().max()
    st.write(f"- The most common category in **{category}** is **{top_value}** with **{top_count} passengers**.")
    
# BAR CHART
elif chart_type == "Bar Chart":
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df_cleaned, x=bar_x, hue=bar_hue, ax=ax)
    ax.set_title(f"{bar_x} by {bar_hue}")
    st.pyplot(fig)
        
    st.subheader("Insights / Analysis")
    group_sizes = df_cleaned.groupby([bar_x, bar_hue]).size()
    largest_group = group_sizes.idxmax()
    largest_count = group_sizes.max()
    st.write(f"- The largest group is **{largest_group}** with **{largest_count} passengers**.")
    
# HISTOGRAM
elif chart_type == "Histogram":
    fig, ax = plt.subplots(figsize=(4, 3))
        
    if df_cleaned[category].dtype == "object" or str(df_cleaned[category].dtype) == "category":
        df_cleaned[category].value_counts().plot(kind="bar", ax=ax, color="purple")
        ax.set_title(f"{category} Frequency")
    else:
        ax.hist(df_cleaned[category].dropna(), bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {category}")
        
    ax.set_xlabel(category)
    ax.set_ylabel("Count")
    st.pyplot(fig)
        
    st.subheader("Insights / Analysis")
    if df_cleaned[category].dtype == "object" or str(df_cleaned[category].dtype) == "category":
        most_common = df_cleaned[category].value_counts().idxmax()
        st.write(f"- The most frequent category in **{category}** is **{most_common}**.")
    else:
        mean_val = df_cleaned[category].mean()
        st.write(f"- The average value of **{category}** is **{mean_val:.2f}**.")

# SIMPLE SCATTER
elif chart_type == "Scatter Plot (Simple)":
    fig, ax = plt.subplots(figsize=(4, 3))
        
    df_numeric = df_cleaned.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == "object" or str(df_numeric[col].dtype) == "category":
            df_numeric[col] = df_numeric[col].astype("category").cat.codes
        
    ax.scatter(df_numeric[x_col], df_numeric[y_col], alpha=0.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    st.pyplot(fig)
        
    st.subheader("Insights / Analysis")
    if df_numeric[[x_col, y_col]].dropna().shape[0] > 1:
        corr_val = df_numeric[[x_col, y_col]].corr().iloc[0, 1]
        st.write(f"- The correlation between **{x_col}** and **{y_col}** is **{corr_val:.2f}**.")
    else:
        st.write("- Not enough data to compute correlation.")
    
# COMPLEX SCATTER
elif chart_type == "Scatter Plot (Complex)":
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=df_cleaned, x=x_col, y=y_col, hue=hue_col, ax=ax)
    ax.set_title(f"{x_col} vs {y_col} colored by {hue_col}")
    st.pyplot(fig)
        
    st.subheader("Insights / Analysis")
    st.write(f"- This chart shows how **{y_col}** varies with **{x_col}** across different **{hue_col}** groups.")
    
# CATEGORICAL HEATMAP (COUNTS)
elif chart_type == "Heatmap (Categorical – Counts)":
    df_clean = df_cleaned[[heat_x, heat_y]].dropna()
    heatmap_data = df_clean.pivot_table(index=heat_y, columns=heat_x, aggfunc="size", fill_value=0)
        
    if heatmap_data.shape[0] > 50 or heatmap_data.shape[1] > 50:
        st.warning("This heatmap has too many categories and may take a long time to render.")
        
    if heatmap_data.empty:
        st.write("No data available for the selected combination.")
    else:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt="d", ax=ax)
        ax.set_title(f"Heatmap: {heat_y} vs {heat_x} (Counts)")
        st.pyplot(fig)
            
        st.subheader("Insights / Analysis")
        max_val = heatmap_data.values.max()
        st.write(f"- The highest count in this table is **{max_val}**, indicating the strongest category combination.")
            
# CATEGORICAL HEATMAP (PERCENTAGES)
elif chart_type == "Heatmap (Categorical Percentages)":
    df_clean = df_cleaned[[heat_x, heat_y]].dropna()
    count_table = df_clean.pivot_table(index=heat_y, columns=heat_x, aggfunc="size", fill_value=0)
        
    if count_table.empty:
        st.write("No data available for the selected combination.")
    else:
        percent_table = count_table.div(count_table.sum(axis=0), axis=1) * 100
            
        if percent_table.shape[0] > 50 or percent_table.shape[1] > 50:
            st.warning("This heatmap has too many categories and may take a long time to render.")
            
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(percent_table, annot=True, cmap="Purples", fmt=".1f", ax=ax)
        ax.set_title(f"Percentage Heatmap: {heat_y} vs {heat_x} (Column %)")
        st.pyplot(fig)
            
        st.subheader("Insights / Analysis")
        max_pct = percent_table.values.max()
        st.write(f"- The highest percentage in this table is **{max_pct:.1f}%**, showing the strongest proportional relationship.")
    
# NUMERICAL CORRELATION HEATMAP
elif chart_type == "Heatmap (Numerical Correlation)":
    numeric_df = df_cleaned.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
        
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Numerical Correlation Heatmap")
    st.pyplot(fig)
        
    st.subheader("Insights / Analysis")
    if "Survived" in corr.columns:
        surv_corr = corr["Survived"].sort_values(ascending=False)
        if len(surv_corr) > 1:
            st.write(f"- Strongest positive correlation with **Survived**: **{surv_corr.index[1]} ({surv_corr.iloc[1]:.2f})**.")
            st.write(f"- Strongest negative correlation with **Survived**: **{surv_corr.index[-1]} ({surv_corr.iloc[-1]:.2f})**.")
    else:
        st.write("- Survived is not numeric in this view, so correlation with survival is not shown.")
    
# 5. SURVIVAL ANALYSIS SUMMARY
st.header("Survival Analysis Highlights (Cleaned Data)")
    
col_sa1, col_sa2 = st.columns(2)
    
with col_sa1:
    st.subheader("Survival by Sex")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df_cleaned, x="Sex", y=df_cleaned["Survived"].astype(int), ax=ax)
    ax.set_ylabel("Survival Rate")
    st.pyplot(fig)

with col_sa2:
    st.subheader("Survival by Pclass")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df_cleaned, x="Pclass", y=df_cleaned["Survived"].astype(int), ax=ax)
    ax.set_ylabel("Survival Rate")
    st.pyplot(fig)
    
    st.subheader("Insights / Analysis")
    st.write(
        "- Females generally show a much higher survival rate than males.\n"
        "- 1st class passengers have the highest survival rate, while 3rd class have the lowest.\n"
        "- These patterns reinforce the impact of gender and socioeconomic status on survival."
    )


st.write("->->-------")
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