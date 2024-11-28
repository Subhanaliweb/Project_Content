from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load your CSV
import pandas as pd
df = pd.read_csv("dataset.csv")

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['class_label'])

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)  # Convert text to TF-IDF features
X = vectorizer.fit_transform(df['text'])
y = df['label_encoded']
print(df['label_encoded'])
print(df['text'])
# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Apply SMOTE to oversample the training data
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

 # Undersampling majority classes
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# # Train classifier on the resampled data
model = RandomForestClassifier()
# model = RandomForestClassifier(class_weight='balanced')
model.fit(X_resampled, y_resampled)

# # Ensemble Models
# model = BalancedRandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# Evaluate on the original test set
y_pred = model.predict(X_test)

# # Classification Report
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))

# # Display the confusion matrix
plt.rcParams['figure.figsize'] = [15, 8]
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=label_encoder.classes_, cmap='Blues')
disp.ax_.tick_params(axis='x', labelrotation=90)
plt.title("Confusion Matrix")
plt.show()

# # Evaluate







# Normal : 0.0013504388926401081
# F1-Score: 0.0013212095944604813

# class_weight='balanced' : 0.004726536124240378

# Smote(Oversampling minority classes) : 0.0013504388926401081

# Undersampling majority classes : 0.16205266711681296

# Ensemble Models : 0.12761647535449022

# neural Network accuracy: 0.1799 - loss: 1.8186 -> Accuracy: [1.7935189008712769, 0.18636056780815125]

#Word2Vec Accuracy: 0.00337609723160027

#Tuned Random Forest: Accuracy: 0.07845369645604004

#Voting Ensemble Accuracy: 0.002025658338960162

#XGBoost Accuracy: 0.0027008777852802163
