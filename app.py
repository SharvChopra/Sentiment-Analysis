# import streamlit as st
# import joblib
# import time

# # --- 1. Define Constants and Filenames ---
# TFIDF_VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
# NAIVE_BAYES_FILE = 'naive_bayes_model.joblib'
# LOGREG_FILE = 'logistic_regression_model.joblib'
# ENSEMBLE_FILE = 'ensemble_model.joblib'


# @st.cache_resource
# def load_models():
#     """
#     Load the vectorizer and all three models (NB, LR, Ensemble).
#     Returns None if files are missing.
#     """
#     try:
#         vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)
#         model_nb = joblib.load(NAIVE_BAYES_FILE)
#         model_logreg = joblib.load(LOGREG_FILE)
#         model_ensemble = joblib.load(ENSEMBLE_FILE)
#         return vectorizer, model_nb, model_logreg, model_ensemble
#     except FileNotFoundError:
#         return None, None, None, None


# vectorizer, model_nb, model_logreg, model_ensemble = load_models()


# def display_result(probability):
#     """
#     Displays a styled metric card for the sentiment.
#     Green for Positive, Red for Negative.
#     """
#     if probability > 0.5:
#         confidence = probability * 100
#         st.success(f"**Positive**\n\nConfidence: {confidence:.1f}%")
#     else:
#         confidence = (1 - probability) * 100
#         st.error(f"**Negative**\n\nConfidence: {confidence:.1f}%")


# st.set_page_config(page_title="Sentiment Analysis Ensemble",
#                    layout="wide", page_icon="🎬")

# st.sidebar.header("ℹ️ About the Models")
# st.sidebar.markdown("""
# **1. Naive Bayes** A probabilistic model. Fast and effective for text, but treats words independently.

# **2. Logistic Regression** A linear model. Assigns a positive/negative "weight" to every word. Very robust.

# **3. 🏆 Ensemble (Voting Classifier)** **The Best Performer.** It combines the predictions of the first two models.
# It uses "Soft Voting" to average the confidence scores, often correcting mistakes made by individual models.
# """)
# st.sidebar.markdown("---")
# st.sidebar.caption("Trained on IMDB 50k Dataset")

# st.title("🎬 IMDB Sentiment Analysis")
# st.markdown("Enter a movie review. The models will analyze the sentiment and show their predictions.")

# if vectorizer is None:
#     st.error("⚠️ **Models not found!**")
#     st.warning(
#         "Please run `python sentiment_analysis_1.py` in your terminal first to train the models and generate the files.")
#     st.stop()

# user_input = st.text_area("Type your movie review here:", height=150,
#                           placeholder="Example: The movie was visually stunning, but the plot was a bit boring...")

# if st.button("Analyze Sentiment", type="primary"):
#     if user_input:
#         if model_nb is None or model_logreg is None or model_ensemble is None:
#             st.error("⚠️ **Models not loaded!** Please restart the app or check the model files.")
#         else:
#             with st.spinner("Analyzing text..."):
#                 time.sleep(0.5)

#                 X_tfidf = vectorizer.transform([user_input])

#                 # --- 2. Predict with ALL models ---
#                 # We get the probability of the "Positive" class (index 1)
#                 prob_nb = model_nb.predict_proba(X_tfidf)[0][1]
#                 prob_lr = model_logreg.predict_proba(X_tfidf)[0][1]
#                 prob_ens = model_ensemble.predict_proba(X_tfidf)[0][1]

#             # --- 3. Display Results ---
#             st.subheader("Model Results Comparison")

#             col1, col2, col3 = st.columns(3)

#             # Column 1: Naive Bayes
#             with col1:
#                 st.markdown("##### 🤖 Naive Bayes")
#                 display_result(prob_nb)

#             # Column 2: Logistic Regression
#             with col2:
#                 st.markdown("##### 📈 Logistic Regression")
#                 display_result(prob_lr)

#             # Column 3: Ensemble (Highlight this one)
#             with col3:
#                 st.markdown("##### 🏆 Ensemble (Combined)")
#                 # Add a little extra visual emphasis
#                 if round(prob_ens) == round(prob_lr) and round(prob_ens) == round(prob_nb):
#                     st.info("All models agree.")
#                 else:
#                     st.info("Ensemble resolved a disagreement!")

#                 display_result(prob_ens)

#     else:
#         st.warning("Please enter some text to analyze.")


import streamlit as st
import joblib
import pandas as pd
import plotly.express as px  # For nice charts

# --- Config ---
st.set_page_config(
    page_title="Advanced Sentiment Lab",
    layout="wide", 
    page_icon="🧪"
)

MODEL_FILE = 'all_models.joblib'

# --- Load Data ---
@st.cache_resource
def load_bundle():
    try:
        data = joblib.load(MODEL_FILE)
        return data
    except FileNotFoundError:
        return None

bundle = load_bundle()

# --- Sidebar ---
st.sidebar.title("🧪 Models Used")
st.sidebar.info(
    "This project implements 7 distinct modeling approaches to compare performance."
)

if bundle:
    # Get all model names from the file
    model_options = list(bundle['models'].keys())
    
    # FIX 1: Set default to 'model_options' so ALL models are selected by default
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        model_options,
        default=model_options 
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Accuracies (Test Set)")
    
    results_df = pd.DataFrame(
        list(bundle['results'].items()), columns=['Model', 'Accuracy']
    )
    results_df['Accuracy'] = results_df['Accuracy'] * 100
    
    st.sidebar.dataframe(
        results_df.sort_values(by='Accuracy', ascending=False)
        .style.format({"Accuracy": "{:.2f}%"})
    )

# --- Main UI ---
st.title("🎬Sentiment Analysis")
st.markdown("### Comparing Base Models vs. Stacking & Voting Ensembles")

if not bundle:
    st.error(f"Model file '{MODEL_FILE}' not found. Please run your training script first.")
    st.stop()

vectorizer = bundle['vectorizer']
models_dict = bundle['models']

# Input
user_input = st.text_area(
    "Enter Movie Review:", 
    height=100,
    placeholder="The cinematography was unique, but the plot fell flat..."
)

if st.button("Run Analysis", type="primary"):
    if user_input:
        
        # Preprocess
        X_input = vectorizer.transform([user_input])

        # Store predictions
        predictions = []

        # Progress bar
        progress_bar = st.progress(0)
        total_selected = len(selected_models) # type: ignore

        for i, model_name in enumerate(selected_models): # type: ignore
            model = models_dict[model_name]

            # Get prediction probability
            # Most models in your kit support predict_proba, but we add a safe fallback
            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(X_input)[0][1]
                except:
                    # Fallback if predict_proba exists but fails (rare)
                    prob = float(model.predict(X_input)[0])
            else:
                prob = float(model.predict(X_input)[0])

            sentiment = "Positive" if prob > 0.5 else "Negative"
            conf = prob if prob > 0.5 else 1 - prob

            predictions.append({
                "Model": model_name,
                "Sentiment": sentiment,
                "Confidence": conf,
                "Raw Score": prob  # For sorting/charting
            })
            progress_bar.progress((i + 1) / total_selected)

        progress_bar.empty()

        # --- Display Results (Grid Layout) ---
        st.subheader("Analysis Results")
        
        # FIX 3: Grid Layout
        # We create 4 columns. We use modulo math (%) to wrap items to the next row.
        cols = st.columns(4)
        
        for idx, res in enumerate(predictions):
            with cols[idx % 4]: # This places prediction 0 in col 0, pred 1 in col 1... pred 4 in col 0
                st.markdown(f"**{res['Model']}**")
                
                if res['Sentiment'] == "Positive":
                    st.success(f"{res['Sentiment']} ({res['Confidence']*100:.1f}%)")
                else:
                    st.error(f"{res['Sentiment']} ({res['Confidence']*100:.1f}%)")
                
                st.markdown("---")

        # --- Comparative Chart ---
        st.subheader("Confidence Visualization")

        pred_df = pd.DataFrame(predictions)

        # Color logic for chart
        pred_df['Color'] = pred_df['Raw Score'].apply(
            lambda x: 'Positive' if x > 0.5 else 'Negative'
        )

        fig = px.bar(
            pred_df,
            x='Model',
            y='Raw Score',
            color='Color',
            color_discrete_map={'Positive': 'green', 'Negative': 'red'},
            range_y=[0, 1],
            title="Model Probability Scores (0=Neg, 1=Pos)"
        )

        # Add a line at 0.5 decision boundary
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Decision Boundary")
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please enter some text to analyze.")