# 🏥 Fall Risk Assessment System

> Machine learning system for proactive patient fall risk classification using hospital incident data — supporting safer clinical environments through data-driven decision making.

---

## 📌 Overview

Hospital patient falls are a significant safety concern, often leading to serious injury and increased care costs. This project applies supervised machine learning to **classify patients into fall-risk categories (High vs. Low/Moderate)** using historical incident data from hospitals spanning 2012–2017.

The system enables healthcare professionals to **proactively identify high-risk patients** — shifting fall prevention from reactive to predictive.

---

## 📄 Research

This project is backed by formal academic research:

| Document | Description |
|---|---|
| [📄 Research Paper](./Fall_Risk_Assessment_Research_Paper.pdf) | Full methodology, results, and analysis |
| [📚 Literature Survey](./FallRiskAssessment_Literature_Survey.pdf) | Review of existing fall-risk detection approaches |

---

## ⚙️ How It Works

### 1. Data Preprocessing (`Clean_Path.py`)
- Handles missing values and null entries across patient records
- Transforms age ranges into numeric representations
- Encodes categorical variables (department, shift, severity) for ML compatibility

### 2. Feature Analysis (`CorelationMtrix.py`)
- Generates a correlation matrix to identify the most predictive features
- Key features analysed: shift timings, department, patient age, incident severity, demographics

### 3. Model Training & Classification (`Medical_Analysis_FallRiskLevel.py`, `Medical_Review_2.py`)
- Trains a **Random Forest Classifier** on the preprocessed dataset
- Classifies each patient record as **High Risk** or **Low/Moderate Risk**
- Handles class imbalance through balanced sampling strategies

### 4. Model Evaluation (`F1_score.py`)
- Evaluates performance using a comprehensive set of metrics

---

## 📊 Results

| Metric | Score |
|---|---|
| **Accuracy** | **85%+** |
| Precision | High (class-balanced) |
| Recall | Optimised for High-Risk detection |
| F1-Score | Reported per class |

The model prioritises **recall for the High-Risk class** — in a healthcare context, a missed high-risk patient is more costly than a false alarm.

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| ML Library | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Model | Random Forest Classifier |

---

## 📁 Repository Structure

```
Fall_Risk_Assessment/
├── Clean_Path.py                          # Data cleaning & preprocessing
├── CorelationMtrix.py                     # Feature correlation analysis
├── Medical_Analysis_FallRiskLevel.py      # Core ML model & classification
├── Medical_Review_2.py                    # Extended model review & analysis
├── F1_score.py                            # Evaluation metrics
├── Fall_Risk_Assessment_Research_Paper.pdf
└── FallRiskAssessment_Literature_Survey.pdf
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the pipeline
```bash
# Step 1: Clean and preprocess the dataset
python Clean_Path.py

# Step 2: Analyse feature correlations
python CorelationMtrix.py

# Step 3: Train and classify
python Medical_Analysis_FallRiskLevel.py

# Step 4: Evaluate model performance
python F1_score.py
```

---

## 💡 Key Insights

- **Shift timing** was among the most correlated features with fall incidents — night shifts showed elevated risk patterns
- **Patient age** combined with **department type** provided strong predictive signal
- Random Forest outperformed baseline classifiers (Logistic Regression, Decision Tree) in recall for the High-Risk class

---

## 🎯 Use Case

This system is designed as a **decision-support tool** for hospital safety teams — not a replacement for clinical judgement. By flagging high-risk patients early, nursing staff can implement targeted fall-prevention protocols (bed alarms, movement assistance, frequent checks).

---

## 👤 Author

**Aarav Walavalkar**  
B.Tech Information Technology, Manipal Institute of Technology  
LinkedIn: https://www.linkedin.com/in/aarav-walavalkar
GitHub: https://github.com/aaravw8804
