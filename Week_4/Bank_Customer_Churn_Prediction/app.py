import streamlit as st
import joblib

# Load the model
filename = 'model.pkl'
loaded_model = joblib.load(filename)

if not hasattr(loaded_model, 'predict'):
    st.error("The loaded object is not a valid model. Please check your 'model.pkl' file.")
    st.stop()

# Streamlit app UI
st.title("Bank Customer Churn Prediction")

with st.form("my_form"):
    gender_input = st.radio("Select Gender", options=["Male", "Female"])
    Gender = 1 if gender_input == "Male" else 0

    Age = st.number_input(label='Age', step=1, format="%d")

    Balance = st.number_input(label='Balance', step=0.1, format="%.2f")

    IsActiveMember = st.radio("Is Active Member?", options=["Yes", "No"])
    IsActiveMember = 1 if IsActiveMember == "Yes" else 0

    EstimatedSalary = st.number_input(label='Estimated Salary', step=1)

    data = [[Gender, Age, Balance, IsActiveMember, EstimatedSalary]]
    submitted = st.form_submit_button("Submit")

if submitted:
    try:
        # Prediction
        prediction = loaded_model.predict(data)[0]
        
        if prediction == 1:
            st.success("The customer **will churn**.")
        else:
            st.success("The customer **will not churn**.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
