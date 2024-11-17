# xgboost_model.py
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# Load Data
df = pd.read_csv("dataset.csv")
df['label_encoded'] = df['class_label'].factorize()[0]

# Vectorize Text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label_encoded']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred, target_names=df['class_label'].unique()))
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=df['class_label'].unique(), cmap='Blues')
disp.ax_.tick_params(axis='x', labelrotation=90)
plt.title("Confusion Matrix - XGBoost")
plt.show()