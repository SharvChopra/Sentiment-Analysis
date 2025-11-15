import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # type: ignore
import json
import numpy as np
import joblib
import re

MAX_LEN = 256

TFIDF_VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
NAIVE_BAYES_FILE = 'naive_bayes_model.joblib'
LOGREG_FILE = 'logistic_regression_model.joblib'
LSTM_MODEL_FILE = 'lstm_model.h5'
KERAS_TOKENIZER_FILE = 'keras_tokenizer.json'


@st.cache_resource
def load_models():
    print("Loading all models...")
    vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)
    model_nb = joblib.load(NAIVE_BAYES_FILE)
    model_logreg = joblib.load(LOGREG_FILE)
    model_lstm = load_model(LSTM_MODEL_FILE)

    with open(KERAS_TOKENIZER_FILE) as f:
        tokenizer_json = f.read()
        keras_tokenizer = tokenizer_from_json(tokenizer_json)

    print("All models loaded.")
    return vectorizer, model_nb, model_logreg, model_lstm, keras_tokenizer


try:
    vectorizer, model_nb, model_logreg, model_lstm, keras_tokenizer = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please run `python train_model.py` first to generate the model files.")
    st.stop()


def preprocess_text_keras(text, tokenizer, max_len):
    """
    Preprocesses raw text for the Keras LSTM model.
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(
        sequence, maxlen=max_len, padding='post', truncating='post')
    return padded_sequence


def display_result(probability):
    """
    Helper function to display a styled result.
    """
    if probability > 0.5:
        confidence = probability * 100
        st.success(f"**Positive** (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - probability) * 100
        st.error(f"**Negative** (Confidence: {confidence:.2f}%)")


st.set_page_config(page_title="Sentiment Model Comparison", layout="wide")
st.title("🎬 IMDB Sentiment Analysis: Model Comparison")
st.markdown("Enter a movie review below to see how three different models (Naive Bayes, Logistic Regression, and LSTM) classify it.")

user_input = st.text_area("Your Movie Review:", height=150,
                          placeholder="This movie was fantastic! The acting was superb and...")

if st.button("Analyze Sentiment", type="primary"):
    if user_input:
        try:

            X_tfidf = vectorizer.transform([user_input])

            X_keras = preprocess_text_keras(
                user_input, keras_tokenizer, MAX_LEN)

            prob_nb = model_nb.predict_proba(
                X_tfidf)[0]  # (prob_neg, prob_pos)

            prob_logreg = model_logreg.predict_proba(
                X_tfidf)[0]  # (prob_neg, prob_pos)

            prob_lstm = model_lstm.predict(X_keras)[0][0]

            st.subheader("Model Predictions:")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Naive Bayes")
                st.markdown("_A fast, statistical 'bag-of-words' model._")
                display_result(prob_nb[1])

            with col2:
                st.markdown("#### Logistic Regression")
                st.markdown(
                    "_A linear 'bag-of-words' model, strong baseline._")
                display_result(prob_logreg[1])

            with col3:
                st.markdown("#### LSTM Neural Network")
                st.markdown(
                    "_A deep learning model that reads words in sequence._")
                display_result(prob_lstm)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("---")
st.markdown(
    "Models trained on the [IMDB dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) using Scikit-learn and Keras.")
