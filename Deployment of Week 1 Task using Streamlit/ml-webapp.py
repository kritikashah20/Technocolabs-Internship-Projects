# Import all the required libraries
import numpy as np
import pickle
import streamlit as st
import pandas as pd


st.write("""
         # Credit Card Default Prediction
This app predicts whether the customer's account will default or not based on some features.
         """)


# Side bar for user inputs
LIMIT_BAL = st.sidebar.text_input("Amount of Credit provided including individual & family credit (in New Taiwanese (NT) dollar)")

education_status = ['graduate school', 'university', 'high school', 'others']
EDUCATION = education_status.index(st.sidebar.selectbox("Select Education", tuple(education_status))) + 1

marital_status = ['Married', 'Single', 'Others']
MARRIAGE = marital_status.index(st.sidebar.selectbox("Marital Status", tuple(marital_status))) + 1

AGE = st.sidebar.text_input("Age (in Years)")

payment_status = [
        "Account started that month with a zero balance and never used any credit",
        "Account had a balance that was paid in full",
        "At least the minimum payment was made, but the entire balance wasn't paid",
        "Payment delay for 1 month",
        "Payment delay for 2 month",
        "Payment delay for 3 month",
        "Payment delay for 4 month",
        "Payment delay for 5 month",
        "Payment delay for 6 month",
        "Payment delay for 7 month",
        "Payment delay for 8 month",   
    ]
PAY_1 = payment_status.index(st.sidebar.selectbox("Last Month Payment Satus", tuple(payment_status))) - 2

BILL_AMT1 = st.sidebar.text_input("Last month Bill Amount (in New Taiwanese (NT) dollar)")
BILL_AMT2 = st.sidebar.text_input("Last 2nd month Bill Amount (in New Taiwanese (NT) dollar)")
BILL_AMT3 = st.sidebar.text_input("Last 3rd month Bill Amount (in New Taiwanese (NT) dollar)")
BILL_AMT4 = st.sidebar.text_input("Last 4th month Bill Amount (in New Taiwanese (NT) dollar)")
BILL_AMT5 = st.sidebar.text_input("Last 5th month Bill Amount (in New Taiwanese (NT) dollar)")
BILL_AMT6 = st.sidebar.text_input("Last 6th month Bill Amount (in New Taiwanese (NT) dollar)")

PAY_AMT1 = st.sidebar.text_input("Amount paid in Last Month (in New Taiwanese (NT) dollar)")
PAY_AMT2 = st.sidebar.text_input("Amount paid in Last 2nd month (in New Taiwanese (NT) dollar)")
PAY_AMT3 = st.sidebar.text_input("Amount paid in Last 3rd month (in New Taiwanese (NT) dollar)")
PAY_AMT4 = st.sidebar.text_input("Amount paid in Last 4th month (in New Taiwanese (NT) dollar)")
PAY_AMT5 = st.sidebar.text_input("Amount paid in Last 5th month (in New Taiwanese (NT) dollar)")
PAY_AMT6 = st.sidebar.text_input("Amount paid in Last 6th month (in New Taiwanese (NT) dollar)")

df = pd.DataFrame([LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6])

if st.button("Predict"):
    # Display the user input features
    st.subheader("User Input Features")
    st.write(df)


    # Load the saved model
    my_model = pickle.load(open('Final_Model.pkl', 'rb'))


    # Make Predictions using the loaded model
    prediction = my_model.predict(df)
    prediction_proba = my_model.predict_proba(df)


    # Display the final predicted results
    st.subheader('Prediction')
    if prediction == 1:
        st.write("The account will be defaulted")
        st.subheader('Prediction Probability')
        st.write("Probability of account being defaulted is {}%.".format(round(prediction_proba * 100, 2)))

    else:
        st.write("The account will not be defaulted")
        st.subheader('Prediction Probability')
        st.write("Probability of account not being defaulted is {}%.".format(round(prediction_proba * 100, 2)))