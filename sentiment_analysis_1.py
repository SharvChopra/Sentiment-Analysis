import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Models ---
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV # To get probabilities from SVM
from sklearn.metrics import accuracy_score

# --- Constants ---
VOCAB_SIZE = 10000
DATASET_FILE = 'IMDB Dataset.csv'
MODEL_FILE = 'all_models.joblib' # Saving all in one dict for cleaner file management

print("--- Starting Advanced Model Training ---")

# 1. Load Data
try:
    print("Loading data...")
    df = pd.read_csv(DATASET_FILE)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    X = df['review']
    y = df['label']
except FileNotFoundError:
    print("Error: IMDB Dataset.csv not found.")
    exit()

# 2. Split Data (Using a smaller subset for speed if needed, but here using full)
# Stratify ensures balanced classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Vectorization (TF-IDF)
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Define Models ---

# A. Base Models
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42) # Restricted depth for speed

# B. SVM (Needs Calibration for Probabilities)
svc = LinearSVC(dual=False, max_iter=1000)
svm_prob = CalibratedClassifierCV(svc) 

# C. PCA/SVD Approach (Pipeline)
# We use TruncatedSVD because standard PCA crashes RAM with sparse text data.
# This is "Latent Semantic Analysis" (LSA)
svd = TruncatedSVD(n_components=100) # Reduce 10,000 features to 100
svd_model = make_pipeline(svd, StandardScaler(with_mean=False), LogisticRegression())

# --- Ensembles ---

# D. Voting Classifier (Soft Voting)
voting_clf = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        ('svm', svm_prob),
        ('rf', rf)
    ],
    voting='soft',
    n_jobs=-1
)

# E. Stacking Classifier
# Uses the output of base models as input for a "Final Estimator" (Logistic Regression)
stacking_clf = StackingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        ('svm', svc), 
        ('rf', rf)
    ],
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)

# --- Dictionary of Models to Train ---
models = {
    "Naive Bayes": nb,
    "Logistic Regression": lr,
    "Random Forest": rf,
    "SVM (Linear)": svm_prob,
    "SVD + LogReg (PCA approach)": svd_model,
    "Voting Ensemble": voting_clf,
    "Stacking Ensemble": stacking_clf
}

results = {}

print(f"\nTraining {len(models)} models... this may take a while.")

for name, model in models.items():
    start_time = time.time()
    print(f"Training {name}...", end=" ")
    
    model.fit(X_train_tfidf, y_train)
    
    # Predict
    pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    
    elapsed = time.time() - start_time
    print(f"Done. Accuracy: {acc*100:.2f}% (Time: {elapsed:.1f}s)")

# --- Save the models---
print("\nSaving all models and vectorizer...")
bundle = {
    "vectorizer": vectorizer,
    "models": models,
    "results": results
}
joblib.dump(bundle, MODEL_FILE)
print(f"Saved to {MODEL_FILE}")