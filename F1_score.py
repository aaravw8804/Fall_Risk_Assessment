import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === Confusion Matrix Plot Function ===
def plot_confusion_matrix(y_true, y_pred, labels=["Low/Moderate", "High"]):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Display the matrix values
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

# === Load and Clean the Data ===
file_path = "C:\\AARAV_Files\\MIT_MANIPAL\\STUDY_MATERIAL\\Sixth_Sem\\DM_Lab_Pics\\hospital-fall-data-2012-2017.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df.fillna("Unknown", inplace=True)

if 'Unnamed: 16' in df.columns:
    df.drop(columns=['Unnamed: 16'], inplace=True)

df['Date of incident'] = pd.to_datetime(df['Date of incident'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

# === Age Transformation ===
def categorize_age(age):
    try:
        if isinstance(age, str):
            numbers = [int(num) for num in re.findall(r'\d+', age)]
            if numbers:
                return f"{min(numbers)}<{max(numbers)}"
            else:
                return 'Unknown'
        age = int(age)
        if age < 13:
            return '1<13'
        elif 13 <= age < 20:
            return '13<20'
        elif 20 <= age < 40:
            return '20<39'
        elif 40 <= age < 60:
            return '40<59'
        elif 60 <= age < 80:
            return '60<79'
        else:
            return '80<100'
    except (ValueError, TypeError):
        return 'Unknown'

if 'Age' in df.columns:
    df['Age range of patient'] = df['Age'].apply(categorize_age)
    df.drop(columns=['Age'], inplace=True)

def convert_age_range_to_float(age_range):
    try:
        if "<" in age_range:
            parts = age_range.split("<")
            return (float(parts[0]) + float(parts[1])) / 2
        elif ">" in age_range:
            return float(age_range[1:])
        else:
            return np.nan
    except ValueError:
        return np.nan

df['Age (float)'] = df['Age range of patient'].apply(lambda x: convert_age_range_to_float(x) if x != 'Unknown' else np.nan)

# === Risk Labeling ===
df['Fall risk (binary)'] = df['Fall risk level'].apply(lambda x: 1 if x == "High" else 0)

# === Attribute Selection ===
selected_columns = [
    'Age (float)', 'Sex', 'Shift', 'Weekday of incident',
    'Hospital department or location of incident', 'Type of injury incurred, if any',
    'Presence of companion at time of incident', 'Location or environment in which the incident ocurred',
    'Reason for incident', 'Whether a fall prevention protocol was implemented',
    'Involvement of medication associated with fall risk', 'Severity of incident', 'Fall risk (binary)'
]

df = df[selected_columns]

# === One-Hot Encoding ===
categorical_columns = [
    'Sex', 'Shift', 'Weekday of incident', 'Hospital department or location of incident',
    'Type of injury incurred, if any', 'Presence of companion at time of incident',
    'Location or environment in which the incident ocurred', 'Reason for incident',
    'Whether a fall prevention protocol was implemented', 'Involvement of medication associated with fall risk',
    'Severity of incident'
]

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("Final columns after selection and encoding:", df.columns)

# === Split Data ===
if 'Fall risk (binary)' in df.columns:
    X = df.drop(columns=['Fall risk (binary)'])
    y = df['Fall risk (binary)']
else:
    print("Error: 'Fall risk (binary)' column not found in the dataset!")
    X, y = None, None

if X is not None and y is not None and not X.empty and not y.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Model Training ===
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # === Prediction and Evaluation ===
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Classification Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n=== Detailed Classification Report ===")
    print(classification_report(y_test, y_pred))

    # === Confusion Matrix Plot ===
    plot_confusion_matrix(y_test, y_pred)

    # === Bar Graph of Class Distribution ===
    plt.figure(figsize=(8, 5))
    df['Fall risk (binary)'].value_counts().plot(kind='bar', color=['blue', 'red'])
    plt.title("Fall Risk Classification Distribution")
    plt.xlabel("Risk Level (0: Low/Moderate, 1: High)")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["Low/Moderate", "High"], rotation=0)
    plt.tight_layout()
    plt.show()

    # === Save Preprocessed File ===
    df.to_csv("Hospital_FallRiskLevel_DM_Lab.csv", index=False)
    print("Data processing and classification complete. Processed file saved as 'Hospital_FallRiskLevel_DM_Lab.csv'")
else:
    print("Error: No data available after preprocessing!")
