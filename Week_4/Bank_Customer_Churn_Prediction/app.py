import pickle
import streamlit as st

# Load the trained model
filename = 'model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

if not hasattr(loaded_model, 'predict'):
    st.error("The loaded object is not a valid model. Please check your 'model.pkl' file.")
    st.stop()

# Streamlit UI
st.title("Churn Prediction")

with st.form("my_form"):
    Gender = st.number_input(label='Gender', step=0.1, format="%.1f")
    Age = st.number_input(label='Age', step=0.01, format="%.6f")
    Balance = st.number_input(label='Balance', step=0.01, format="%.6f")
    IsActiveMember = st.number_input(label='IsActiveMember', step=1)
    EstimatedSalary = st.number_input(label='EstimatedSalary', step=1)

    data = [[Gender, Age, Balance, IsActiveMember, EstimatedSalary]]
    submitted = st.form_submit_button("Submit")

if submitted:
    try:
        prediction = loaded_model.predict(data)[0]
        st.write(f"Customer will churn (1 = Yes, 0 = No): {prediction}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
