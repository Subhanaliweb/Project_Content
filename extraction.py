import os
import pandas as pd

# Since the script is in the same directory as the dataset folders, set data_path to "."
data_path = "."

data = []

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".txt"):
            # Get the folder structure as the label
            folder_path = os.path.relpath(root, data_path)
            if folder_path == ".":
                continue  # Skip processing the current directory where the script is
            class_label = folder_path.replace("/", "_")  # Combine folder names for labels
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                data.append({"class_label": class_label, "text": text})

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)
