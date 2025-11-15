import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.text import Tokenizer  # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense  # pyright: ignore[reportMissingImports]
import numpy as np
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

VOCAB_SIZE = 10000
MAX_LENGTH = 256
EMBEDDING_DIM = 16

TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
NAIVE_BAYES_FILE = 'naive_bayes_model.joblib'
LOGISTIC_REGRESSION_FILE = 'logistic_regression_model.joblib'

LSTM_MODEL_PATH = 'lstm_model.h5'
KERAS_TOKENIZER_PATH = 'keras_tokenizer.json'

# Load Raw IMDB DATASET
train_ds, test_ds = tfds.load(
    'imdb_reviews', split=['train', 'test'], as_supervised=True)


def process_text(dataset):
    texts = []
    labels = []

    for text, label in dataset.as_numpy_iterator():
        texts.append(text.decode('utf-8'))
        labels.append(label)
    return texts, np.array(labels)


train_texts, train_labels = process_text(train_ds)
test_texts, test_labels = process_text(test_ds)

print(
    f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples.")

# --- 3. Train Bag-of-Words Models (Naive Bayes & Logistic Regression) ---
print("\n--- Training Bag-of-Words Models ---")

# --- 3a. Create TF-IDF Features ---
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, stop_words='english')

X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)

print(f"TF-IDF feature matrix shape (train): {X_train_tfidf.shape}")

print("Training Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, train_labels)
nb_pred = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(test_labels, nb_pred)
print(f"Naive Bayes Test Accuracy: {nb_accuracy * 100:.2f}%")

print("Training Logistic Regression model...")
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_tfidf, train_labels)
logreg_pred = logreg_model.predict(X_test_tfidf)
logreg_accuracy = accuracy_score(test_labels, logreg_pred)
print(f"Logistic Regression Test Accuracy: {logreg_accuracy * 100:.2f}%")

# --- 3d. Save Sklearn Models ---
print("Saving TF-IDF vectorizer and sklearn models...")
joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
joblib.dump(nb_model, NAIVE_BAYES_FILE)
joblib.dump(logreg_model, LOGISTIC_REGRESSION_FILE)

# --- 4. Train LSTM Neural Network ---
print("\n--- Training LSTM Model ---")

print("Tokenizing and padding sequences for LSTM...")
keras_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<UNK>")
keras_tokenizer.fit_on_texts(train_texts)

X_train_seq = keras_tokenizer.texts_to_sequences(train_texts)
X_test_seq = keras_tokenizer.texts_to_sequences(test_texts)

X_train_pad = pad_sequences(
    X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH,
                           padding='post', truncating='post')

# --- 4b. Build the LSTM Model ---
print("Building the Keras LSTM model...")
model_lstm = Sequential([
    # Input_dim is VOCAB_SIZE, output_dim is EMBEDDING_DIM
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM,
              input_length=MAX_LENGTH),

    # Use a Bidirectional LSTM layer for better context understanding
    Bidirectional(LSTM(64)),  # 64 units

    # A hidden layer
    Dense(16, activation='relu'),

    # Output layer
    Dense(1, activation='sigmoid')
])

model_lstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(model_lstm.summary())

print("Training the LSTM model...")
history = model_lstm.fit(
    X_train_pad,
    train_labels,
    epochs=5,  # Reduced epochs for faster training, 10 is also good
    batch_size=512,
    validation_data=(X_test_pad, test_labels),
    verbose=1
)

loss, accuracy = model_lstm.evaluate(X_test_pad, test_labels)
print(f"\nLSTM Test Accuracy: {accuracy*100:.2f}%")

print(f"Saving LSTM model to {LSTM_MODEL_PATH}...")
model_lstm.save(LSTM_MODEL_PATH)

print(f"Saving Keras Tokenizer to {KERAS_TOKENIZER_PATH}...")
tokenizer_json = keras_tokenizer.to_json()
with open(KERAS_TOKENIZER_PATH, 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)

print("--- All Training Complete ---")
