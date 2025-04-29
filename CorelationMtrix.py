import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
file_path = "C:\\AARAV_Files\\MIT_MANIPAL\\STUDY_MATERIAL\\Sixth_Sem\\DM_Lab_Pics\\hospital-fall-data-2012-2017.csv"
df = pd.read_csv(file_path)

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Convert 'Date of incident' to datetime (if needed)
if 'Date of incident' in df.columns:
    df['Date of incident'] = pd.to_datetime(df['Date of incident'], errors='coerce')

# Convert categorical variables to numeric using Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le  # Store encoder for reference

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix using Matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
cax = ax.matshow(correlation_matrix, cmap="coolwarm")

# Add color bar
plt.colorbar(cax)

# Set axis labels
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.columns)))

ax.set_xticklabels(correlation_matrix.columns, rotation=90)
ax.set_yticklabels(correlation_matrix.columns)

plt.title("Correlation Matrix - Hospital Fall Data", pad=20)
plt.show()
