import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered", initial_sidebar_state="expanded")

# Main title and description
st.title("Credit Card Fraud Detection Model 💳")
st.markdown("""
    *Detect whether a credit card transaction is fraudulent or legitimate.*
    This Web page allows you to input transaction details and predicts whether it's a fraud based on a machine learning model.
""")

# Sidebar for more interaction
st.sidebar.header("Model Information")

# Upload CSV file for dataset replacement
st.sidebar.subheader("Upload CSV Dataset")
uploaded_csv = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Initialize model and data variables
data = None  # Data variable to hold dataset
model = None  # Model variable to hold trained model
last_input_features = None  # Variable to hold last input features for prediction

def preprocess_data(data):
    """
    Preprocess the data by undersampling legitimate transactions
    and preparing features and labels.
    """
    # Separate legitimate and fraudulent transactions
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # Check for class imbalance
    if len(fraud) == 0:
        st.error("No fraudulent transactions found in the dataset.")
        return None, None

    # Undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    # Split data into features (X) and labels (y)
    X = balanced_data.drop(columns="Class", axis=1)
    y = balanced_data["Class"]
    
    return X, y

if uploaded_csv:
    try:
        # Attempt to read the CSV file
        data = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV file loaded successfully.")

        # Check if 'Class' column exists
        if 'Class' not in data.columns:
            st.error("The uploaded dataset does not contain the required 'Class' column.")
            st.stop()

        # Display total counts of legitimate and fraudulent transactions
        total_legit = data[data.Class == 0].shape[0]
        total_fraud = data[data.Class == 1].shape[0]

        # Display the counts in color-coded boxes
        st.markdown(f"<div style='background-color: #0B5345; color: white; padding: 10px; border-radius: 5px; margin-bottom: 5px;'><b>Total Legitimate Transactions:</b> {total_legit}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: #8B0000; color: white; padding: 10px; border-radius: 5px; margin-bottom: 5px;'><b>Total Fraudulent Transactions:</b> {total_fraud}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.sidebar.error(f"Error loading CSV file: {e}")
else:
    # Fallback to default CSV if no file is uploaded
    data = pd.read_csv('creditcard.csv')
    st.sidebar.info("Using default dataset (creditcard.csv).")

# Check if data is loaded
if data is not None and 'Class' in data.columns:
    # Preprocess the data
    X, y = preprocess_data(data)

    if X is not None and y is not None:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

        # Train logistic regression model
        model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
        model.fit(X_train, y_train)

        # Evaluate model performance
        train_acc = accuracy_score(model.predict(X_train), y_train)
        test_acc = accuracy_score(model.predict(X_test), y_test)

        # Show model accuracy in sidebar
        st.sidebar.markdown(f"""
        - *Training Accuracy:* {train_acc * 100:.2f}%  
        - *Test Accuracy:* {test_acc * 100:.2f}%  
        """)

# User input form in the sidebar
st.sidebar.subheader("Input Transaction Details")
input_features = st.sidebar.text_input('Enter all features separated by commas')

# Create a button to submit input and get prediction
submit = st.sidebar.button("Predict")

# Display predictions
if submit:
    if input_features:  # Check if input is not empty
        try:
            # Convert input to an array of floats
            input_data = np.array([float(i) for i in input_features.split(',')]).reshape(1, -1)

            # Ensure the model is trained before making predictions
            if model is not None:
                # Make prediction
                prediction = model.predict(input_data)

                # Display results with colored boxes and minimal spacing
                if prediction[0] == 0:
                    st.markdown("<div style='background-color: #0B5345; color: white; padding: 10px; border-radius: 5px; margin-top: 5px;'><b>The transaction is legitimate.</b></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color: #8B0000; color: white; padding: 10px; border-radius: 5px; margin-top: 5px;'><b>The transaction is fraudulent.</b></div>", unsafe_allow_html=True)

                last_input_features = input_features  # Store last input features for future reference
            else:
                st.error("Model is not trained yet. Please upload a dataset first.")
        except ValueError:
            st.error("Invalid input. Please ensure all features are numeric and separated by commas.")
    else:
        st.error("Please enter transaction features to proceed.")

# Add footer or more description
st.markdown("---")
st.markdown("""
*How does it work?*  
- The logistic regression model is trained on a balanced dataset using undersampling techniques.  
- Input your transaction features to get a prediction on whether the transaction is legitimate or fraudulent.  
""")