import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the pre-trained model
model = load_model('model.h5')

# Load word_index from the saved JSON file
with open('word_index.json', 'r') as fp:
    word_index = json.load(fp)

# Create a reverse word_index for decoding predictions
reverse_word_index = {v: k for k, v in word_index.items()}

# Create a tokenizer
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")  # Adjust based on your original tokenizer settings
tokenizer.word_index = word_index  # Set the word_index to the loaded one

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove stopwords (you can reuse your stopwords removal logic here)
    # Tokenize and convert to sequences using the tokenizer you defined
    seq = tokenizer.texts_to_sequences([text])
    # Pad the sequence
    padded = pad_sequences(seq, maxlen=20, padding='post', truncating='post')
    return padded

# Streamlit app
st.title('Review Sentiment Prediction')
st.subheader('This APP to predict sentiment Postive or Negatives Review Restaurants')

# Input text box for user to enter a review
user_input = st.text_input('Enter a review:')

if user_input:
    # Preprocess the user input
    preprocessed_input = preprocess_text(user_input)
    
    # Make a prediction
    prediction = model.predict(preprocessed_input)
    
    # Determine sentiment based on the prediction
    if prediction > 0.5:
        sentiment = 'Positive Review'
    else:
        sentiment = 'Negative Review'
    
    st.write(f'Sentiment: {sentiment}')


# add article
st.sidebar.title('More Article')
st.sidebar.markdown(f"[Improving Employee Retention by Predicting Employee Attrition Using Machine Learning](https://medium.com/@aqilafadiamariana/improving-employee-retention-by-predicting-employee-attrition-using-machine-learning-f576bea204d8)")
st.sidebar.markdown(f"[Investigate Hotel Business using Data Visualization](https://medium.com/@aqilafadiamariana/investigate-hotel-business-using-data-visualization-cef104723962)")
st.sidebar.markdown(f"[Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning](https://medium.com/@aqilafadiamariana/predict-customer-personality-to-boost-marketing-campaign-by-using-machine-learning-79368c2dc87d)")
st.sidebar.markdown(f"[Classification of Skin Cancer model using Tensorflow](https://medium.com/@aqilafadiamariana/classification-of-skin-cancer-model-using-tensorflow-19ff2c000087)")
st.sidebar.markdown(f"[Predict Credit Scores by Enhancing Precision Function](https://medium.com/@aqilafadiamariana/predict-credit-scores-by-enhancing-precision-function-a0bb104528d)")