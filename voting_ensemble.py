# voting_ensemble.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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

# Define Models
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Voting Classifier
voting_model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
voting_model.fit(X_train, y_train)

# Evaluate
y_pred = voting_model.predict(X_test)

accuracy = voting_model.score(X_test, y_test)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred, target_names=df['class_label'].unique()))
disp = ConfusionMatrixDisplay.from_estimator(voting_model, X_test, y_test, display_labels=df['class_label'].unique(), cmap='Blues')
disp.ax_.tick_params(axis='x', labelrotation=90)
plt.title("Confusion Matrix - Voting Ensemble")
plt.show()
