import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
file_path = "C:\\AARAV_Files\\MIT_MANIPAL\\STUDY_MATERIAL\\Sixth_Sem\\DM_Lab_Pics\\hospital-fall-data-2012-2017.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Step 1: Handle missing values
df.fillna("Unknown", inplace=True)

# Step 2: Drop unnecessary columns
if 'Unnamed: 16' in df.columns:
    df.drop(columns=['Unnamed: 16'], inplace=True)

# Step 3: Convert date column
df['Date of incident'] = pd.to_datetime(df['Date of incident'], errors='coerce')

# Step 4: Convert year
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

# Step 5: Categorize age
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

# Step 6: Convert age range to float
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

# Step 7: Binary classification for fall risk
df['Fall risk (binary)'] = df['Fall risk level'].apply(lambda x: 1 if x == "High" else 0)

# Step 8: Select relevant features
selected_columns = [
    'Age (float)', 'Sex', 'Shift', 'Weekday of incident',
    'Hospital department or location of incident', 'Type of injury incurred, if any',
    'Presence of companion at time of incident', 'Location or environment in which the incident ocurred',
    'Reason for incident', 'Whether a fall prevention protocol was implemented',
    'Involvement of medication associated with fall risk', 'Severity of incident', 'Fall risk (binary)'
]

df = df[selected_columns]

# Step 9: One-Hot Encoding
categorical_columns = [
    'Sex', 'Shift', 'Weekday of incident', 'Hospital department or location of incident',
    'Type of injury incurred, if any', 'Presence of companion at time of incident',
    'Location or environment in which the incident ocurred', 'Reason for incident',
    'Whether a fall prevention protocol was implemented', 'Involvement of medication associated with fall risk',
    'Severity of incident'
]

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Step 10: Train-test split
X = df.drop(columns=['Fall risk (binary)'])
y = df['Fall risk (binary)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 12: Predict & evaluate
y_pred = rf_model.predict(X_test)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

# Step 13: Save processed dataset
df.to_csv("Hospital_FallRiskLevel_DM_Lab.csv", index=False)
print("Processed data saved as 'Hospital_FallRiskLevel_DM_Lab.csv'")

# Step 14: Bar plot for binary fall risk counts
plt.figure(figsize=(8, 4))
counts = df['Fall risk (binary)'].value_counts()
plt.bar(['Low/Moderate (0)', 'High (1)'], counts, color=['blue', 'red'])
plt.title("Fall Risk Level Distribution")
plt.xlabel("Fall Risk Level")
plt.ylabel("Number of Incidents")
plt.tight_layout()
plt.show()

# Step 15: Display classification report as table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.set_title('Classification Report', fontsize=14, fontweight='bold')
table_data = [report_df.columns.tolist()] + report_df.reset_index().values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')
table.scale(1.2, 1.2)
plt.tight_layout()
plt.show()
