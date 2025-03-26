# 1. Import packages
import numpy as np
import pandas as pd
import pickle
import os,requests,json
import streamlit as st
# Load model
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl','rb'))
    return model
model = load_model()
# 2. Build the streamlit app components
# Structure the input widgets
# Start with a title widget
st.title("Loan Approval Status Application 💰")
st.header("This is application to help predicting the loan approval status of client.")
st.header("Please input the value correctly.")
# Dictionary for categorical data
home_ownership = {'MORTGAGE':0.0,'OTHER':1.0,'OWN':2.0,'RENT':3.0}
grade_loan = {'A':0.0,'B':1.0,'C':2.0,'D':3.0,'E':4.0,'F':5.0,'G':6.0}

# Follow by input widgets
# Input widgets using a form for better organization
with st.form(key="user_inputs"):
    col1, col2 = st.columns(2)  # Create two columns for better layout

    with col1:
        person_age = st.slider("Age (Years)", 0, 80, step=1)
        person_income = st.slider("Income (USD)", 0.0, 6000000.0, step=1.0)
        person_home_ownership = home_ownership[st.selectbox("Homeownership Status", list(home_ownership.keys()))]
        person_emp_length = st.slider("Employment Length (Years)", 0, 50, step=1)
        loan_grade = grade_loan[st.selectbox("Loan (Risk) Grade", list(grade_loan.keys()))]

    with col2:
        loan_amnt = st.slider("Loan Amount", 0.0, 3500.0, step=1.0)
        loan_int_rate = st.slider("Loan Interest Rate", 0.0, 50.0, step=1.0)
        loan_percent_income = st.slider("Loan Percent Income (%)", 0.0, 1.0, step=0.01)
        cb_person_cred_hist_length = st.slider("Credit History Length (Years)", 0, 30, step=1)

    submit = st.form_submit_button("Predict Loan Status")

if submit:
    # Structure the input into a 2D numpy array with (batch, feature) dimensions
    input_np = np.array([person_age, person_income, person_home_ownership, person_emp_length, loan_grade,
                         loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length])
    input_np = np.expand_dims(input_np, axis=0)

    # Use the input data to make a prediction
    label_map = ['Fail', 'Success']
    y_pred = model.predict(input_np)

    st.subheader("Prediction Result:")
    if y_pred[0] == 1:
        st.success(f"Loan Status: {label_map[y_pred[0]]}")
    else:
        st.error(f"Loan Status: {label_map[y_pred[0]]}")

    st.write("---")  # Add a separator for better visual clarity

    # Display Input Values for User Review
    st.subheader("Input Values:")
    st.write(f"**Age:** {person_age} years")
    st.write(f"**Income:** ${person_income:,.2f}")
    st.write(f"**Homeownership:** {list(home_ownership.keys())[list(home_ownership.values()).index(person_home_ownership)]}")
    st.write(f"**Employment Length:** {person_emp_length} years")
    st.write(f"**Loan Grade:** {list(grade_loan.keys())[list(grade_loan.values()).index(loan_grade)]}")
    st.write(f"**Loan Amount:** ${loan_amnt:,.2f}")
    st.write(f"**Interest Rate:** {loan_int_rate}%")
    st.write(f"**Percent Income:** {loan_percent_income * 100}%")
    st.write(f"**Credit History Length:** {cb_person_cred_hist_length} years")