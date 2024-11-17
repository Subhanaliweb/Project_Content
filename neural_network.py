# neural_network.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd

# Load Data
df = pd.read_csv("dataset.csv")
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['class_label'])

# Vectorize Text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = to_categorical(df['label_encoded'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate
y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test.argmax(axis=1), y_pred, target_names=label_encoder.classes_))

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

disp = ConfusionMatrixDisplay.from_predictions(y_test.argmax(axis=1), y_pred, display_labels=label_encoder.classes_, cmap='Blues')
disp.ax_.tick_params(axis='x', labelrotation=90)
plt.title("Confusion Matrix - Neural Network")
plt.show()