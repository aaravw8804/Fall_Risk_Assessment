import pandas as pd
from tabulate import tabulate

# === Load Preprocessed Data ===
file_path = r"C:\Aarav-Files\STUDY_MATERIAL\Sixth_Sem\DM_Lab_Pics\cleaned_fall_data_2.xlsx"
df = pd.read_excel(file_path)
df.columns = [col.strip() for col in df.columns]
df.fillna("Unknown", inplace=True)

# === Column Mappings ===
df.rename(columns={
    "Weekday of incident": "Weekday",
    "Shift": "Shift",
    "Hospital department or location of incident": "Hospital department",
    "Age range of patient": "Age",
    "Type of injury incurred, if any": "Type of injury",
    "Presence of companion at time of incident": "Presence of companion",
    "Location or environment in which the incident ocurred": "Location",
    "Fall risk level": "Fall risk",
    "Reason for incident": "Reason",
    "Whether a fall prevention protocol was implemented": "Prevention protocol",
    "Involvement of medication associated with fall risk": "Medication",
    "Severity of incident": "Severity",
    "Sex": "Gender",
}, inplace=True)

# === Age Group Standardization ===
def standardize_age(age):
    if age == "<1":
        return "<1 years"
    if age == "1<13":
        return "<19 years"
    elif age == "13<19":
        return "<19 years"
    elif "≥ 90" in age or age.startswith("90"):
        return "≥90 years"
    match = pd.Series(age).str.extract(r'(\d+)<(\d+)')
    if match.isnull().any(axis=None):
        return age
    low = int(match.iloc[0, 0])
    if low < 20:
        return "<19 years"
    elif low < 30:
        return "20–29 years"
    elif low < 40:
        return "30–39 years"
    elif low < 50:
        return "40–49 years"
    elif low < 60:
        return "50–59 years"
    elif low < 70:
        return "60–69 years"
    elif low < 80:
        return "70–79 years"
    elif low < 90:
        return "80–89 years"
    else:
        return "≥90 years"

df["Age"] = df["Age"].astype(str).apply(standardize_age)

# === Convert Binary Target ===
df["Fall risk"] = df["Fall risk"].apply(lambda x: "High" if x == "High" else "Low-Moderate")

# === Predictors to Include ===
predictors = [
    "Age", "Gender", "Shift", "Weekday", "Hospital department", "Type of injury",
    "Presence of companion", "Location", "Reason", "Prevention protocol",
    "Medication", "Severity", "Fall risk"
]

# === Function to Generate Table ===
def generate_summary_table(df, predictors, target="Fall risk"):
    total = len(df)
    summary = []

    for var in predictors:
        rows = []
        categories = df[var].value_counts().index
        for cat in categories:
            all_count = df[df[var] == cat].shape[0]
            high_count = df[(df[var] == cat) & (df[target] == "High")].shape[0]
            low_count = df[(df[var] == cat) & (df[target] != "High")].shape[0]
            row = [
                cat,
                f"{all_count} ({round(all_count/total*100)}%)",
                f"{high_count} ({round(high_count/all_count*100)}%)" if all_count else "0 (0%)",
                f"{low_count} ({round(low_count/all_count*100)}%)" if all_count else "0 (0%)"
            ]
            rows.append(row)
        summary.append([var, "", "", ""])  # section header
        summary.extend(rows)
    df_summary = pd.DataFrame(summary, columns=["Predictor", "All", "High", "Low-Moderate"])
    return df_summary

# === Generate and Display Table ===
summary_df = generate_summary_table(df, predictors)
print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

# === Optionally Save as CSV or Excel ===
summary_df.to_csv("fall_risk_summary_table.csv", index=False)


