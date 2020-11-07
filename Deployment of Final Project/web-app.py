# import all the required libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# main function
def main():
    st.write("""
    # Toxic Comments Classification
    This app predicts the various categories a comment belongs to""")

    comment = st.text_input("Enter your comment here", )

    if st.button("Predict"):
        # display user entered comment
        st.write(comment)

        # load the saved model
        f = open("final_model.pkl", "rb")
        model = pickle.load(f)
        v = pickle.load(f)

        lis = []
        for i in range(6):
            lis.append(model[i].predict_proba(v.transform([comment]))[:, 1])
        list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


        # Display the final predicted results
        st.subheader('Prediction')
        for i in range(6):
            st.subheader(list[i])
            st.write(str(lis[i]))
        st.write('----')
if __name__ == '__main__':
    main()