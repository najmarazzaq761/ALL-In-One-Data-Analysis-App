
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error

# add title
st.title("All-in-One Data Analysis")
st.subheader("This is a simple Data Analysis App developed by Najma Razzaq")

# load the data
dataset_options=["titanic","flights","fmri","iris","diamonds"]
selected_dataset=st.selectbox("select a dataset" ,dataset_options)
if selected_dataset=="titanic":
  df=sns.load_dataset("titanic")
elif selected_dataset=="flights":
  df=sns.load_dataset("flights")
elif selected_dataset=="fmri":
  df=sns.load_dataset("fmri")
elif selected_dataset=="iris":
  df=sns.load_dataset("iris")
elif selected_dataset=="diamonds":
  df=sns.load_dataset("diamonds")

# upload dataset
uploaded_file=st.file_uploader("upload  your dataset",type=["xlsx","csv"])
if uploaded_file is not None:
  df=pd.read_excel(uploaded_file)

# display data
st.write(df)
st.write("number of rows",df.shape[0])
st.write("number of columns",df.shape[1])
# display column names with their datatypes
st.write("column names and their datatypes",df.dtypes)

# print null values
if df.isnull().sum().sum()>0:
  st.write("Null values : ",df.isnull().sum().sort_values(ascending=False))
else:
  st.write("No Null Values")

# display summary statistics
st.write("summary statistics:",df.describe())

# Impute missing values
if df.isnull().sum().sum() > 0:
        st.write("### Handling Missing Values")

        # Select columns to impute
        columns_with_nan = df.columns[df.isnull().any()].tolist()
        impute_columns = st.multiselect("Select columns to impute", columns_with_nan)
        
        # Imputation options
        impute_method = st.selectbox("Imputation method", ["Mean", "Median", "Mode","Custom Value"])
        
        for col in impute_columns:
            if impute_method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif impute_method == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif impute_method == "Mode":
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    st.error(f"No mode found for column {col}")
            elif impute_method == "Custom Value":
                custom_value = st.number_input(f"Enter custom value for {col}", value=0.0)
                df[col].fillna(custom_value, inplace=True)
        
st.write(f"### Data after imputing columns using {impute_method}")
st.write(df.isnull().sum())

# Data Visualization/Plotting
st.write("## Data Visualization")
x_axis = st.selectbox("Select x_axis", df.columns)
plot_type = st.selectbox("Select plot type", ["line", "scatter", "hist", "kde", "box", "violin", "count", "bar"])

# For plot types that require y-axis selection
if plot_type in ["line", "scatter", "kde", "box", "violin"]:
    y_axis = st.selectbox("Select y_axis", df.columns)

# Ask if user wants to add a hue for certain plot types
add_hue = False
if plot_type in ["scatter", "box", "violin", "count"]:
    add_hue = st.checkbox("Do you want to add a hue?")
    if add_hue:
        hue = st.selectbox("Select hue", df.columns)

# Plot based on user selection
if plot_type == "line":
    st.line_chart(df[[x_axis, y_axis]])
elif plot_type == "scatter":
    fig, ax = plt.subplots()
    if add_hue:
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
    else:
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)
elif plot_type == "hist":
    fig, ax = plt.subplots()
    df[x_axis].plot(kind="hist", ax=ax)
    st.pyplot(fig)
elif plot_type == "kde":
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)
elif plot_type == "bar":
    st.bar_chart(df[x_axis].value_counts())
elif plot_type == "count":
    fig, ax = plt.subplots()
    if add_hue:
        sns.countplot(x=x_axis, hue=hue, data=df, ax=ax)
    else:
        sns.countplot(x=x_axis, data=df, ax=ax)
    st.pyplot(fig)
elif plot_type == "box":
    fig, ax = plt.subplots()
    if add_hue:
        sns.boxplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
    else:
        sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)
elif plot_type == "violin":
    fig, ax = plt.subplots()
    if add_hue:
        sns.violinplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
    else:
        sns.violinplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

st.write("### Heatmap")
# Filter the DataFrame to include only numeric columns
numeric_df = df.select_dtypes(include=[float, int])
# Calculate the correlation matrix
correlation = numeric_df.corr()
# Plot the heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(correlation, cmap="BrBG", annot=True, ax=ax)
# Display the plot in the Streamlit app
st.pyplot(fig)

# feature Engineering
st.write("### Feature Engineering")
# Multiselect for dropping columns
columns_to_drop = st.multiselect("Select columns to drop", df.columns)
if columns_to_drop:
    df.drop(columns=columns_to_drop, inplace=True)
st.write(f"### Data after dropping columns: {', '.join(columns_to_drop)}")
st.write(df.head())
st.write(df.shape)

# Variable selection for prediction
# Ask user to select target variable (y)
target_variable = st.selectbox("Select the target variable", df.columns)

# Encode selected columns using LabelEncoder for both X and y
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create X (input features) and y (target variable)
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Display X and y
st.subheader("X (Input Features):")
st.write(X)
st.subheader("y (Target Variable):")
st.write(y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

st.subheader("Train-Test Split")
st.write("X_train shape:")
st.write(X_train.shape)
st.write("X_test shape:")
st.write(X_test.shape)
st.write("y_train shape:")
st.write(y_train.shape)
st.write("y_test shape:")
st.write(y_test.shape)

# Model selection
st.write("## Model Training")
problem_type = st.selectbox("Select the problem type", ["Classification", "Regression"])

if problem_type == "Classification":
    model_options = ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "Gradient Boosting Classifier", "Naive Bayes", "SVC"]
    selected_model = st.selectbox("Select a model", model_options)

    if selected_model == "Logistic Regression":
        model = LogisticRegression()
    elif selected_model == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif selected_model == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
    elif selected_model == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
    elif selected_model == "Naive Bayes":
        model = GaussianNB()
    elif selected_model == "SVC":
        model = SVC()

elif problem_type == "Regression":
    model_options = ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor", "Gradient Boosting Regressor", "SVR"]
    selected_model = st.selectbox("Select a model", model_options)

    if selected_model == "Linear Regression":
        model = LinearRegression()
    elif selected_model == "Random Forest Regressor":
        model = RandomForestRegressor()
    elif selected_model == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
    elif selected_model == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
    elif selected_model == "SVR":
        model = SVR()

# Train the selected model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Display results
if problem_type == "Classification":
    accuracy = accuracy_score(y_test, predictions)
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
elif problem_type == "Regression":
    mse = mean_squared_error(y_test, predictions)
    st.write("### Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")




