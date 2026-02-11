# 🚢 Titanic Data Explorer - Streamlit App

A complete end‑to‑end Exploratory Data Analysis (EDA) and Data Cleaning workflow built with Streamlit.  
This interactive app allows users to explore the Titanic dataset, visualize patterns, clean the data, and uncover survival insights.

---

## 📌 Features

### **1. Data Acquisition**

- Loads the Titanic dataset directly from a public GitHub source.
- Displays raw data, column descriptions, and dataset structure.

### **2. Initial EDA (Before Cleaning)**

- Summary statistics
- Data types
- Missing value analysis
- Raw distributions and visualizations
- Correlation heatmap
- Categorical heatmaps

This section helps diagnose data quality issues before cleaning.

---

## 🧹 **3. Data Cleaning**

The app performs a full cleaning pipeline suitable for machine learning:

- **Age** → filled with median
- **Embarked** → filled with mode
- **Cabin** → dropped (too many missing values)
- **Pclass, Survived** → converted to categorical
- **Feature Engineering:**
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone = 1 if FamilySize == 1 else 0`
- Drops `SibSp` and `Parch` after feature creation
- Displays the cleaned dataset and missing‑value summary

This produces a modeling‑ready dataset.

---

## 📊 **4. Post‑Cleaning EDA & Visualizations**

All visualizations use the cleaned dataset.

### Supported Charts:

- Pie Chart
- Bar Chart
- Histogram
- Scatter Plot (Simple)
- Scatter Plot (Complex with hue)
- Heatmap (Categorical — Counts)
- Heatmap (Categorical — Percentages)
- Heatmap (Numerical Correlation)

Each chart includes an **Insights / Analysis** section that explains the patterns shown.

---

## 🛟 **5. Survival Analysis**

The app highlights key survival patterns:

- Survival by Sex
- Survival by Passenger Class
- Insights summarizing gender and socioeconomic effects

---

## 🧠 **6. Modeling‑Ready Output**

The cleaned dataset includes:

- No missing values (except intentionally preserved fields)
- Engineered features
- Correct data types
- Removed high‑missingness and redundant columns

This makes it ready for:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Any ML pipeline you want to build next

---

## ▶️ **How to Run the App**

### **1. Install dependencies**

```bash
pip install streamlit pandas seaborn matplotlib
```
