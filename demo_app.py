"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Group3.


    Description: This file is used to launch a minimal streamlit web
	application. 
"""
# Streamlit dependencies
import streamlit as st
from PIL import Image
# model dependencies 
import pickle
import zipfile
import os
# Data dependencies
import contractions
from nltk.stem import WordNetLemmatizer
from cleaner import handle_weblinks, clean_data, tokenize, transform
import pandas as pd
import emoji

# Random forest model is too large to push to GitHub, so we push a zipped version and unzip it on github
# file = "resources/team3_rand_for.zip"
# path = 'resources/team3_rand_for.pkl'
# if not os.path.exists(path):
#     with zipfile.ZipFile(file, 'r') as rf_model:
#         rf_model.extractall('resources')

# The main function where we will build the actual app


def main():

    """Tweet Classifier App with Streamlit """

    # Creates a main title and subhead on your page - # these are static across all pages
    
    img = Image.open("img1.jpeg")
    img2 = Image.open("img.png")
    img3 = Image.open("img_1.png")
    img4 = Image.open("img_2.png")
    img5 = Image.open("img_3.png")
    img6 = Image.open("img_3.png")
    pos = Image.open("img_3.png")
    neg = Image.open("img_3.png")
    neu = Image.open("img_3.png")
    fact = Image.open("img_3.png")
    
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.title("Meta Analytics")
        st.image([img2, img3], width = 500,)
        st.subheader("What We Do As a Company")
        st.markdown("""
            We specialize in helping small and large companies build machine learning solutions
            that would help grow their business. Since our inception, we have succesfully carried
            and completed over 5000 projects for high profile clients. Here at Meta Analytics, we 
            offer a wide range of products and services that would suite your every business needs.
            some of our product offerings are:

                - Consultations

                - Investment Advisory

                - Machine Learning

                - Data Analytics

                - Project Management

                - Employee Training

            We also help firms seamlessly transition from on site computing to a cloud based 
            adoption framework.
            Visit out website today at:
        """)

        # You can read a markdown file from supporting resources folder
        st.info("www.purplemetaanalytics.co.za/offerings")
        st.image(img4, width=700,)

    # Building out the predication page
    if selection == "Prediction":
        st.title("Meta Analytics")
        st.subheader("Climate change tweet classification")
        st.image([img5, img6], width=500,)
        st.info("To Make Predictions, Type in the Tweets Using the Box Below")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text")

        models = ['logistic_regression', 'naive_bayes']
        selection = st.selectbox("Choose Model to make prediction with hit the classify button when done", models)
        
        if selection == 'logistic_regression':
            predictor = pickle.load(open("team3_log_reg.pkl", "rb"))
        elif selection == 'naive_bayes':
            predictor = pickle.load(open("team3_naive_bayes.pkl", "rb"))
        # elif selection == 'Random_Forest':
        #     predictor = pickle.load(open("resources/team3_rand_for.pkl","rb"))
        # elif selection == 'support vector':
        #     predictor = pickle.load(open("resources/team3_svc.pkl","rb"))
        else:
            predictor = pickle.load(open("team3_log_reg.pkl", "rb"))

        if st.button("Classify"):

            df = pd.DataFrame({'text': [tweet_text]})
            df['text'] = df['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
            df['text'] = [' '.join(map(str, l)) for l in df['text']]   
            # handle weblinks, if any
            df['text'] = df['text'].apply(handle_weblinks) 

            # clean the data
            df['text'] = df['text'].apply(clean_data)

            # tokenize the data
            df['text'] = df['text'].apply(tokenize)

            tweet = [" ".join(i) for i in df.text]  
            prediction = predictor.predict(tweet)
            prediction_map = {1: 'Positive', 2: 'Factual', -1: 'Negative', 0: 'Neutral'}
            pred = prediction_map[prediction[0]]

            # When model has successfully run, will print prediction
            st.success("Text Categorized as a {} Tweet".format(pred))
            if pred == 'Positive':
                st.image(pos, width = 300,)
            elif pred == 'Negative':
                st.image(neg, width = 300,)
            elif pred == 'Neutral':
                st.image(neu, width = 300,)
            elif pred == 'Factual':
                st.image(fact, width = 300,)
            
        st.image(img, width=600,)

# Required to let Streamlit instantiate our web app.


if __name__ == '__main__':

    main()
