import streamlit as st
import joblib
import time

# --- 1. Define Constants and Filenames ---
TFIDF_VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
NAIVE_BAYES_FILE = 'naive_bayes_model.joblib'
LOGREG_FILE = 'logistic_regression_model.joblib'
ENSEMBLE_FILE = 'ensemble_model.joblib'

# --- 2. Load All Models and Preprocessors ---


@st.cache_resource
def load_models():
    """
    Load the vectorizer and all three models (NB, LR, Ensemble).
    Returns None if files are missing.
    """
    try:
        vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)
        model_nb = joblib.load(NAIVE_BAYES_FILE)
        model_logreg = joblib.load(LOGREG_FILE)
        model_ensemble = joblib.load(ENSEMBLE_FILE)
        return vectorizer, model_nb, model_logreg, model_ensemble
    except FileNotFoundError:
        return None, None, None, None


# Load the models when the app starts
vectorizer, model_nb, model_logreg, model_ensemble = load_models()

# --- 3. Helper Function for Displaying Results ---


def display_result(probability):
    """
    Displays a styled metric card for the sentiment.
    Green for Positive, Red for Negative.
    """
    if probability > 0.5:
        confidence = probability * 100
        st.success(f"**Positive**\n\nConfidence: {confidence:.1f}%")
    else:
        confidence = (1 - probability) * 100
        st.error(f"**Negative**\n\nConfidence: {confidence:.1f}%")

# --- 4. Streamlit UI Layout ---


# Page Config
st.set_page_config(page_title="Sentiment Analysis Ensemble",
                   layout="wide", page_icon="🎬")

# Sidebar
st.sidebar.header("ℹ️ About the Models")
st.sidebar.markdown("""
**1. Naive Bayes** A probabilistic model. Fast and effective for text, but treats words independently.

**2. Logistic Regression** A linear model. Assigns a positive/negative "weight" to every word. Very robust.

**3. 🏆 Ensemble (Voting Classifier)** **The Best Performer.** It combines the predictions of the first two models. 
It uses "Soft Voting" to average the confidence scores, often correcting mistakes made by individual models.
""")
st.sidebar.markdown("---")
st.sidebar.caption("Trained on IMDB 50k Dataset")

# Main Page Title
st.title("🎬 IMDB Sentiment Analysis")
st.markdown("Enter a movie review. The models will analyze the sentiment and show their predictions.")

# Error Handling: Check if models exist
if vectorizer is None:
    st.error("⚠️ **Models not found!**")
    st.warning(
        "Please run `python train_model.py` in your terminal first to train the models and generate the files.")
    st.stop()

# Input Area
user_input = st.text_area("Type your movie review here:", height=150,
                          placeholder="Example: The movie was visually stunning, but the plot was a bit boring...")

# Analyze Button
if st.button("Analyze Sentiment", type="primary"):
    if user_input:
        # Verify all models are loaded before proceeding
        if model_nb is None or model_logreg is None or model_ensemble is None:
            st.error("⚠️ **Models not loaded!** Please restart the app or check the model files.")
        else:
            with st.spinner("Analyzing text..."):
                # Artificial delay for visual effect (optional)
                time.sleep(0.5)

                # --- 1. Preprocess ---
                # Transform user text into numbers using the trained TF-IDF vectorizer
                X_tfidf = vectorizer.transform([user_input])

                # --- 2. Predict with ALL models ---
                # We get the probability of the "Positive" class (index 1)
                prob_nb = model_nb.predict_proba(X_tfidf)[0][1]
                prob_lr = model_logreg.predict_proba(X_tfidf)[0][1]
                prob_ens = model_ensemble.predict_proba(X_tfidf)[0][1]

            # --- 3. Display Results ---
            st.subheader("Model Results Comparison")

            col1, col2, col3 = st.columns(3)

            # Column 1: Naive Bayes
            with col1:
                st.markdown("##### 🤖 Naive Bayes")
                display_result(prob_nb)

            # Column 2: Logistic Regression
            with col2:
                st.markdown("##### 📈 Logistic Regression")
                display_result(prob_lr)

            # Column 3: Ensemble (Highlight this one)
            with col3:
                st.markdown("##### 🏆 Ensemble (Combined)")
                # Add a little extra visual emphasis
                if round(prob_ens) == round(prob_lr) and round(prob_ens) == round(prob_nb):
                    st.info("All models agree.")
                else:
                    st.info("Ensemble resolved a disagreement!")

                display_result(prob_ens)

    else:
        st.warning("Please enter some text to analyze.")

