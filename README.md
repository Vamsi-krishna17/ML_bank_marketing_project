# Bank Marketing ML Project

This project predicts whether a client will subscribe to a term deposit.

## Preprocessing & Scaling

### Why Robust Scaler?
During preprocessing, numerical features are scaled using **RobustScaler**.
Unlike `StandardScaler` which scales features based on the mean and standard deviation (making it sensitive to outliers), `RobustScaler` scales using the **Interquartile Range (IQR = Q1–Q3)**. This makes it robust and insensitive to outliers, handling potential outliers in the dataset gracefully.

## Model Selection & Deployment

### Why Logistic Regression with ADASYN?
The final deployed model is **Logistic Regression with ADASYN** (Adaptive Synthetic Sampling). The primary reasons for this choice are:

1. **Business Objective (Highest Recall):**
   The main business goal is to minimize missed subscribers, which corresponds to minimizing **False Negatives (Type 2 Error)**. LR with ADASYN achieves the highest **Recall**, ensuring we identify as many potential subscribers as possible.

2. **Prediction Latency (Speed):**
   Logistic Regression has a significantly faster prediction latency per sample compared to more complex tree-based models (like Random Forest or XGBoost). This low latency makes it highly suitable for real-time deployment at scale.
