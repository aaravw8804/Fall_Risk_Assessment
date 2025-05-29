import pandas as pd
import re

# Input Excel file path
file_path = r"C:\Aarav-Files\STUDY_MATERIAL\Sixth_Sem\DM_Lab_Pics\combined_hospital_fall_data.xlsx"

# Load Excel data
df = pd.read_excel(file_path)
df.columns = [col.strip() for col in df.columns]

# Mapping actual column names to required 13 attributes
column_mapping = {
    "Age range of patient": "Age",
    "Sex": "Gender",
    "Shift": "Shift",
    "Weekday of incident": "Weekday",
    "Hospital department or location of incident": "Hospital department",
    "Type of injury incurred, if any": "Type of injury",
    "Presence of companion at time of incident": "Presence of companion",
    "Location or environment in which the incident ocurred": "Location",
    "Reason for incident": "Reason",
    "Whether a fall prevention protocol was implemented": "Prevention protocol",
    "Involvement of medication associated with fall risk": "Medication",
    "Severity of incident": "Severity",
    "Fall risk level": "Fall risk level"
}

df = df.rename(columns=column_mapping)
required_columns = list(column_mapping.values())
df = df[required_columns]

# Updated age range conversion logic
def convert_age_range(age_str):
    if not isinstance(age_str, str):
        return "Unknown"
    age_str = age_str.strip()

    if age_str == "< 1":
        return "<=19"
    elif age_str == "1<13":
        return "<=19"
    elif age_str == "13<19" or age_str == "13<20":
        return "<=19"
    elif age_str == "60<70":
        return "60-69"
    elif age_str == "70<80":
        return "70-79"
    elif age_str == "80<90":
        return "80-89"
    elif age_str == "≥ 90":
        return ">=90"

    match = re.match(r"(\d+)<(\d+)", age_str)
    if match:
        lower = int(match.group(1))
        # Group ages into 20-year ranges: 20-39, 40-59
        base = (lower // 20) * 20
        return f"{base}-{base + 19}"

    return "Unknown"

df["Age"] = df["Age"].apply(convert_age_range)

# Save the cleaned output
output_path = r"C:\Aarav-Files\STUDY_MATERIAL\Sixth_Sem\DM_Lab_Pics\cleaned_fall_data_2.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Cleaned data saved successfully to:\n{output_path}")
