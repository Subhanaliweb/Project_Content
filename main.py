from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("dataset.csv")

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['class_label'])

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(texts):
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    # Use the [CLS] token representation as the sentence embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

# Extract BERT embeddings
print("Extracting BERT embeddings...")
X = get_bert_embeddings(df['text'].tolist())
y = df['label_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate on the test set
y_pred = clf.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Display the confusion matrix
plt.rcParams['figure.figsize'] = [15, 8]
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=label_encoder.classes_, cmap='Blues')
disp.ax_.tick_params(axis='x', labelrotation=90)
plt.title("Confusion Matrix")
plt.show()

# Evaluate
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))