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
st.title("🎬IMDB Sentiment Analysis")
st.markdown(
    "Enter a movie review. The models will analyze the sentiment and show their predictions.")

if not bundle:
    st.error(
        f"Model file '{MODEL_FILE}' not found. Please run your training script first.")
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
        total_selected = len(selected_models)  # type: ignore

        for i, model_name in enumerate(selected_models):  # type: ignore
            model = models_dict[model_name]

            # Get prediction probability
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

        cols = st.columns(4)

        for idx, res in enumerate(predictions):
            with cols[idx % 4]:
                st.markdown(f"**{res['Model']}**")

                if res['Sentiment'] == "Positive":
                    st.success(
                        f"{res['Sentiment']} ({res['Confidence']*100:.1f}%)")
                else:
                    st.error(
                        f"{res['Sentiment']} ({res['Confidence']*100:.1f}%)")

                st.markdown("---")

        # --- Comparative Chart ---
        st.subheader("Confidence Visualization")

        pred_df = pd.DataFrame(predictions)

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

        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Decision Boundary")

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please enter some text to analyze.")
