import streamlit as st
import pandas as pd
import pickle  # Use pickle to load the model
from pre_processor import PreProcessor

# Load your trained model
with open('best_model.pkl', 'rb') as file:
    data= pickle.load(file)


model=data['best_model']
# Streamlit UI
st.title('Prediction App')

# File uploader allows user to add their own CSV or Excel
uploaded_file = st.file_uploader("Upload your input CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the uploaded file according to its format
    if uploaded_file.type == "text/csv":
        dataset = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        dataset = pd.read_excel(uploaded_file)

    # Dataset-specific processing
    pd.set_option('future.no_silent_downcasting', True)
    dataset['total_calls'] = dataset['total_day_calls'] + dataset['total_eve_calls'] + dataset['total_night_calls']
    dataset['total_minutes'] = dataset['total_day_minutes'] + dataset['total_eve_minutes'] + dataset['total_night_minutes']
    dataset['total_charges'] = dataset['total_day_charge'] + dataset['total_eve_charge'] + dataset['total_night_charge']
    dataset.drop(['total_day_calls', 'total_eve_calls', 'total_night_calls', 'total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_day_charge', 'total_eve_charge', 'total_night_charge'], axis=1, inplace=True)
    dataset['churn'] = dataset['churn'].replace({'no': 0, 'yes': 1})
    churn_column = dataset['churn']
    dataset.drop('churn', axis=1, inplace=True)
    dataset['churn'] = churn_column

    # Pre-processing
    pre_processor = PreProcessor(dataset)
    binary_encode_columns = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
    one_hot_encode_columns = []
    ordinal_encode_columns = []
    scaling_columns = ['account_length', 'number_vmail_messages', 'total_intl_minutes', 'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls', 'total_calls', 'total_minutes', 'total_charges']

    pre_processor.remove_nans().\
      remove_duplicates().\
        scale(scaling_columns).\
          binary_encode(binary_encode_columns).\
            onehot_encode(one_hot_encode_columns).\
              ordinal_encode(ordinal_encode_columns).\
                oversample()
    pre_processor.calculate_feature_importance()
    pre_processor.remove_features_by_importance(0.00)
    pre_processor.split_features_labels()
    X = pre_processor.X
    y = pre_processor.y
    print(X)
    # Display the processed dataframe
    st.write("Processed Data Preview:")
    st.dataframe(X) 

    # Predict button
    if st.button('Predict'):
        predictions = model.predict(X)
        
        # Append predictions to the dataframe for display
        # Create a copy of X to display to avoid modifying the original dataframe
        X_display = X.copy()  # Create a copy of X to modify
        X_display['churn'] = predictions  # Append the predictions as a new column
        
        # Displaying the updated dataframe with predictions
        st.write("Data Preview with Predictions:")
        st.dataframe(X_display)