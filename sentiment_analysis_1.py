import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

VOCAB_SIZE = 10000  # Max features for TF-IDF
DATASET_FILE = 'IMDB Dataset.csv'  # The file you downloaded from Kaggle

TFIDF_VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
NAIVE_BAYES_FILE = 'naive_bayes_model.joblib'
LOGREG_FILE = 'logistic_regression_model.joblib'
ENSEMBLE_FILE = 'ensemble_model.joblib'

print("--- Starting Model Training (Ensemble Approach) ---")

# --- 2. Load and Preprocess Data ---
print(f"Loading dataset from {DATASET_FILE}...")
try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_FILE}'")
    print("Please download the dataset from Kaggle and place it in this folder.")
    exit()

df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Initializing individual models...")
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)

print("Training Ensemble Model (Voting Classifier)...")
# 'soft' voting averages the probabilities of the inputs
ensemble_model = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr)],
    voting='soft'
)
ensemble_model.fit(X_train_tfidf, y_train)

# We train these separately just so we can show their individual scores to the user
print("Training individual models for comparison...")
nb.fit(X_train_tfidf, y_train)
lr.fit(X_train_tfidf, y_train)

# --- 7. Evaluate Models ---
print("\n--- Evaluation Results ---")

nb_pred = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"1. Naive Bayes Accuracy:       {nb_acc * 100:.2f}%")

lr_pred = lr.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"2. Logistic Regression Accuracy: {lr_acc * 100:.2f}%")

ens_pred = ensemble_model.predict(X_test_tfidf)
ens_acc = accuracy_score(y_test, ens_pred)
print(f"3. Ensemble (Combined) Accuracy: {ens_acc * 100:.2f}%")

if ens_acc > nb_acc and ens_acc > lr_acc:
    print("\nResult: The Ensemble model outperformed the individual models!")
else:
    print("\nResult: The Ensemble model performed similarly to the best individual model.")

# --- 8. Save Models ---
print("Saving all models...")
joblib.dump(vectorizer, TFIDF_VECTORIZER_FILE)
joblib.dump(nb, NAIVE_BAYES_FILE)
joblib.dump(lr, LOGREG_FILE)
joblib.dump(ensemble_model, ENSEMBLE_FILE)

print("--- All Training Complete ---")
