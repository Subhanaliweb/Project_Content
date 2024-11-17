# rf_hyperparameter_tuning.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, target_names=df['class_label'].unique()))


accuracy = grid_search.score(X_test, y_test)
print("Accuracy:", accuracy)

disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=df['class_label'].unique(), cmap='Blues')
disp.ax_.tick_params(axis='x', labelrotation=90)

plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()
