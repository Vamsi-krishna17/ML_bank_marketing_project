# 🏦 Bank Marketing — Term Deposit Subscription Prediction

An end-to-end **Machine Learning project** that predicts whether a bank client will subscribe to a **term deposit**, built as part of the Machine Learning Unique Project at **Innomatics Research Labs**.

🔗 **Live App:** [Add your Streamlit Cloud link here]
📂 **Dataset Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [Problem Statement](#-problem-statement)
- [Dataset Overview](#-dataset-overview)
- [Project Workflow](#-project-workflow)
- [Data Cleaning](#-data-cleaning)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Model Building](#-model-building)
- [Model Comparison](#-model-comparison)
- [Final Model — Why Logistic Regression + ADASYN?](#-final-model--why-logistic-regression--adasyn)
- [Streamlit App](#-streamlit-app)
- [Tech Stack](#-tech-stack)
- [How to Run Locally](#-how-to-run-locally)
- [Project Structure](#-project-structure)
- [Conclusion](#-conclusion)

---

## 📋 Business Problem

A Portuguese retail bank runs **telephone-based marketing campaigns** to sell **term deposit subscriptions**. Multiple calls are often made to the same client, which is **expensive and time-consuming**.

The bank wants to use historical campaign data to identify **which clients are most likely to subscribe**, so call-center resources are spent wisely and clients aren't unnecessarily disturbed.

> **Objective:** Predict whether a client will subscribe to a term deposit (`yes` / `no`) using their demographic profile, past campaign interactions, and economic indicators.

---

## 🎯 Problem Statement

> Given historical data of bank clients and their interactions during past telephone marketing campaigns, build a **supervised binary classification model** that predicts whether a client will subscribe to a term deposit — target variable `y`.

**Key Assumptions:**
- Data is representative of the real client population
- Phone calls are the sole marketing channel in this dataset
- Economic indicators (`emp.var.rate`, `euribor3m`, etc.) are external and not controllable
- Class imbalance (majority = "no") is handled using **SMOTE** and **ADASYN**
- `default` column was dropped — too many "unknown" values, near-zero variance
- **Minimizing False Negatives (missed subscribers) is the core business priority**

---

## 🔍 Dataset Overview

| Detail | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| File | `bank-additional-full.csv` |
| Rows | 41,188 |
| Columns | 21 original → 20 after dropping `default` |
| Delimiter | Semicolon (`;`) |
| Target Variable | `y` (yes / no) |

**Class Distribution:**

| Class | Count | Percentage |
|---|---|---|
| no  | 36,548 | ~88% |
| yes |  4,640 | ~12% |

⚠️ **Significant class imbalance** — addressed using SMOTE and ADASYN during model training.

![Dataset Overview](screenshots/dataset_overview.png)

---

## 🗺️ Project Workflow

```
Business Problem → Problem Statement → Data Collection
        ↓
Dataset Overview → Feature Description → Data Cleaning
        ↓
EDA (Univariate, Bivariate, Multivariate)
        ↓
Preprocessing (Split → OHE → Scaling → Label Encoding)
        ↓
Model Building (6 Models × 3 Balancing Variants)
        ↓
Model Comparison → Best Model Selection → Hyperparameter Tuning
        ↓
Streamlit Deployment (Predict Page)
```

---

## 🧹 Data Cleaning

- **No missing (null) values** found in the dataset
- **`default` column dropped** — ~20% "unknown" values, near-zero variance, no predictive value
- **"unknown" values retained** in other categorical columns — treated as a valid category since they may carry signal
- **No outlier removal/clipping** — every record represents a real client call; extreme values are genuine. **RobustScaler** (IQR-based) handles outliers during scaling instead

![Data Cleaning](screenshots/data_cleaning.png)

---

## 📊 Exploratory Data Analysis

Performed across three levels, each with **interactive column and plot-type selection** in the Streamlit app:

### Univariate Analysis
- Numerical: Histogram, Box Plot, KDE, Violin Plot — with Mean, Median, Std Dev, Skewness, Kurtosis, and IQR-based outlier detection
- Categorical: Vertical/Horizontal Bar, Pie Chart — with value counts and percentages

### Bivariate Analysis
- Categorical vs Target — Grouped Bar (%), Stacked % Bar, Count Bar
- Numerical vs Target — Overlapping Histogram, KDE Comparison, Box/Violin by Target
- Numerical vs Numerical — Scatter, Hexbin, Line plots with Pearson correlation

### Multivariate Analysis
- Correlation Heatmap — identifies multicollinearity among economic indicators (`emp.var.rate`, `euribor3m`, `nr.employed`)
- Pair Plot — relationships across multiple numerical features, colored by target

![EDA Example](screenshots/eda_example.png)

**Key Insight:** `emp.var.rate`, `cons.price.idx`, `euribor3m`, and `nr.employed` are **highly correlated** macro-economic indicators — handled via Robust Scaling and regularization during modeling.

---

## ⚙️ Data Preprocessing

| Step | Method |
|---|---|
| Train-Test Split | 80% / 20%, `random_state=42` |
| Categorical Encoding | One-Hot Encoding (`pd.get_dummies`) on 9 columns |
| Column Alignment | Test set reindexed to match train columns exactly |
| Numerical Scaling | **RobustScaler** (IQR-based, outlier-resistant) |
| Target Encoding | LabelEncoder — `no` = 0, `yes` = 1 |

**Why RobustScaler?**
Since no outlier clipping was performed, the dataset contains genuine extreme values (e.g., long call durations). RobustScaler scales based on the **interquartile range (Q1–Q3)**, making it naturally insensitive to outliers — unlike StandardScaler or MinMaxScaler.

**Final Shapes:**

| Variable | Shape |
|---|---|
| X_train_sc | (32950, 60) |
| X_test_sc | (8238, 60) |
| y_train_enc | (32950,) |
| y_test_enc | (8238,) |

![Preprocessing](screenshots/preprocessing.png)

---

## 🤖 Model Building

**6 Algorithms** trained:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. Gradient Boosting
5. AdaBoost

**3 Balancing Variants** for each model:

| Variant | Description |
|---|---|
| **Base** | Original imbalanced training data (0 = 29,245 \| 1 = 3,705) |
| **SMOTE** | Synthetic Minority Oversampling — balances classes (0 = 29,245 \| 1 = 29,245) |
| **ADASYN** | Adaptive Synthetic Sampling — focuses on harder-to-classify minority samples (0 = 29,245 \| 1 = 28,707) |

→ **15 model combinations** trained and evaluated.

**Evaluation Metrics:** Accuracy, Precision, **Recall**, F1-Score, ROC-AUC, Confusion Matrix, **Prediction Latency** (measured over 1000 predictions per model).

---

## ⚖️ Model Comparison

### Why Recall Over ROC-AUC?

In an **imbalanced dataset**, Accuracy is misleading — a model predicting "no" for everyone still scores ~88% accuracy while being completely useless.

**ROC-AUC** measures overall separability and is robust to imbalance — but it doesn't directly reflect the business cost of errors.

For this project, a **False Negative (Type 2 Error)** means the bank **misses a real subscriber** — direct revenue loss. A **False Positive** just means an extra unnecessary call. The costs are asymmetric, so:

> ✅ **Recall is the primary selection metric** — it directly measures how many real subscribers the model successfully identifies.

### Full Comparison Table (Selected Results)

| Model | Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | Missed Subscribers (FN) |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | Base (Tuned) | 0.920 | 0.67 | 0.5583 | 0.6098 | **0.9484** | 414 |
| XGBoost | Base | 0.913 | 0.63 | 0.5551 | 0.5911 | 0.9448 | 420 |
| Gradient Boosting | SMOTE | 0.892 | 0.51 | 0.8267 | 0.6339 | 0.9427 | 162 |
| Gradient Boosting | ADASYN | 0.883 | 0.49 | 0.8417 | 0.6194 | 0.9387 | 148 |
| Logistic Regression | SMOTE | 0.861 | 0.44 | 0.8759 | 0.5892 | 0.9367 | 116 |
| **Logistic Regression** | **ADASYN (Tuned)** | **0.847** | **0.42** | **0.8995** | **0.5711** | **0.9355** | **94** ✅ |

![Model Comparison](screenshots/model_comparison.png)
![Recall Comparison](screenshots/recall_comparison.png)
![Latency Comparison](screenshots/latency_comparison.png)

---

## 🏆 Final Model — Why Logistic Regression + ADASYN?

| Criteria | Gradient Boosting (Best ROC-AUC) | **Logistic Regression + ADASYN (Deployed)** |
|---|---|---|
| ROC-AUC | 0.9484 | 0.9355 |
| **Recall** | 0.5583 | **0.8995** ✅ |
| Missed Subscribers | 414 / 935 | **94 / 935** ✅ |
| Prediction Latency | Higher (tree ensemble) | **Lowest — real-time capable** ✅ |
| Interpretability | Low | **High — clear feature coefficients** ✅ |

### Decision Rationale

> Although **Gradient Boosting (Base, Tuned)** achieved the highest ROC-AUC (0.9484), it **misses 414 out of 935 actual subscribers** — a major revenue loss for the bank.
>
> **Logistic Regression with ADASYN balancing** achieves **Recall = 0.8995**, missing only **94 subscribers**. ADASYN focuses on the hardest-to-classify minority samples near the decision boundary, helping the model generalize better on real subscribers.
>
> It is also the **fastest model in prediction latency**, making it ideal for **real-time scoring** of large client lists — and its coefficients remain interpretable for business stakeholders.
>
> **Hyperparameter tuning** was performed via `RandomizedSearchCV` (scoring = Recall, 5-fold CV) — the tuned model was deployed only because it improved Recall further over the base ADASYN model.

### Confusion Matrix — Final Model

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | True Negative | False Positive |
| **Actual Yes** | False Negative (94) | True Positive (841) |

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## 🖥️ Streamlit App

An **11-page interactive web application**:

| Page | Description |
|---|---|
| 🏠 Home | Project overview and workflow |
| 📋 Business Problem | Domain context and challenges |
| 🎯 Problem Statement | ML formulation and assumptions |
| 📦 Data Collection | Dataset source and background |
| 🔍 Dataset Overview | Shape, dtypes, missing values, target distribution |
| 📑 Feature Description | All 20 features documented with filters |
| 🧹 Data Cleaning | Missing values, unknowns, outlier strategy |
| 📊 EDA | Interactive Univariate / Bivariate / Multivariate analysis |
| ⚙️ Preprocessing | Step-by-step pipeline with before/after visuals |
| 🤖 Model Building | Full comparison across Base/SMOTE/ADASYN + latency |
| 🔮 **Predict** | Live prediction — enter client details, get instant result |

### Predict Page

Users enter:
- **Client Info:** age, job, marital status, education, housing/personal loan
- **Campaign Details:** contact type, month, day, call duration, number of contacts, previous outcome
- **Economic Indicators:** auto-filled with latest dataset values (adjustable)

→ Returns **YES / NO** prediction with subscription probability and visual breakdown.

![Predict Page](screenshots/predict_page.png)

---

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML Libraries:** Scikit-learn, XGBoost, imbalanced-learn (SMOTE, ADASYN)
- **Model Persistence:** Joblib
- **Deployment:** Streamlit, Streamlit Community Cloud

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/bank-marketing-ml.git
cd bank-marketing-ml

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-run preprocessing and model training
python preprocessing.py
python models.py

# 5. Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```
Bank Marketing/
│
├── app.py                      # Streamlit application (11 pages)
├── preprocessing.py             # Data cleaning, encoding, scaling, LR base model
├── models.py                    # Trains 6 models × 3 variants, tuning, latency
├── bank-additional-full.csv     # Dataset (UCI ML Repository)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
└── models/
    ├── preprocessed_data.joblib   # Scaler, label encoder, train columns, scaled data
    ├── best_model.joblib          # Final deployed model — LR (ADASYN, Tuned)
    ├── best_model_info.joblib     # Best model metadata, reason, latency
    ├── all_metrics.joblib         # Metrics for all 15+ model combinations
    └── latency_results.joblib     # Prediction latency for every model
```

---

## ✅ Conclusion

This project demonstrates a **complete, business-driven ML pipeline** — from problem framing through deployment:

- Identified and addressed **class imbalance** using SMOTE and ADASYN
- Trained and compared **6 algorithms across 3 balancing strategies** (15 combinations)
- Selected the final model based on **business-aligned metrics (Recall)** rather than blindly optimizing for ROC-AUC or Accuracy
- **Logistic Regression + ADASYN (Tuned)** was deployed — achieving **89.95% Recall**, missing only 94 of 935 actual subscribers, with the **lowest prediction latency** for real-time use
- Delivered as a fully interactive **Streamlit web application** with a live prediction interface

> 💡 **Key Takeaway:** The "best" model isn't always the one with the highest accuracy or ROC-AUC — it's the one that best serves the **actual business objective**. Here, minimizing missed subscribers (False Negatives) directly translates to **maximized revenue capture** for the bank.

---

## 👤 Author

**Vamsi Krishna**
Innomatics Research Labs — Machine Learning Unique Project

🔗 [LinkedIn](#) | [GitHub](#) | [Portfolio](#)
