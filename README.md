#  Customer Churn Prediction using Machine Learning

Predicting customer churn for telecom companies using Logistic Regression and Random Forest with a full ML pipeline.

---

##  1. Problem Statement

Customer churn is a major issue for subscription-based businesses.
Acquiring new customers is more expensive than retaining existing ones.

**Goal:**
Build a machine learning model to predict whether a customer will churn so the company can take preventive actions.

---

##  2. Dataset

We used the **Telco Customer Churn Dataset**.

![Image](https://editor.analyticsvidhya.com/uploads/89646table%20info%201.png)

![Image](https://opengraph.githubassets.com/8b6dea8c5db10229b2d0fe1643ce53e0fffe1dccaa41877f6f7731bcce7a628d/mohammedali9810/Telco-Customer-Churn-Prediction)

![Image](https://www.researchgate.net/publication/357302836/figure/tbl1/AS%3A1104466824695809%401640336819693/Dataset-of-telecommunication-customer-churn.png)

![Image](https://editor.analyticsvidhya.com/uploads/468605.png)

Dataset contains:

* Customer demographics
* Subscription details
* Payment methods
* Service usage
* Target variable → **Churn**

Total rows ≈ 7,000 customers.

---

##  3. Exploratory Data Analysis (EDA)

Key insights discovered:

* Customers with **month-to-month contracts** have the highest churn.
* Customers with **low tenure (<12 months)** churn more.
* Customers without **Tech Support or Online Security** churn more.
* Customers using **Electronic Check** payment method show higher churn.
* **Fiber optic users** have higher churn probability.

These insights guided feature engineering and model selection.

---

##  4. Data Preprocessing

Used a full **Pipeline + ColumnTransformer** to avoid data leakage.

### Numerical Features

* Imputation using median
* Standard scaling

### Categorical Features

* One-Hot Encoding
* Handle unknown categories safely

This ensures consistent preprocessing during training and prediction.

---

##  5. Models Used

### 1️⃣ Logistic Regression

* Used as baseline model
* Applied class imbalance handling with `class_weight="balanced"`

### 2️⃣ Random Forest

* Used for comparison
* Captures nonlinear patterns

---

##  6. Model Performance

| Model               | ROC-AUC  | Recall (Churn=1) | Precision |
| ------------------- | -------- | ---------------- | --------- |
| Logistic Regression | **0.84** | **0.79**         | 0.51      |
| Random Forest       | 0.83     | 0.51             | **0.63**  |

### Final Model Choice:

Logistic Regression was chosen because detecting churn customers is more important than minimizing false alarms.

---

##  7. Feature Importance Insights

Top features increasing churn:

* Fiber optic internet
* Electronic check payment
* Streaming services
* Multiple lines
* Paperless billing

Top features reducing churn:

* Long tenure
* One-year / Two-year contracts
* Online security services
* Customers with dependents

---

##  8. Business Recommendations

Based on model insights:

* Offer discounts for long-term contracts.
* Improve fiber internet service quality.
* Provide free tech support trial for new customers.
* Encourage auto-pay instead of electronic check.
* Focus retention efforts on customers with tenure < 12 months.

---

##  9. Manual Testing

Model was tested on random customers from test set to verify prediction behavior and ensure correct pipeline preprocessing.

---

##  10. Model Deployment

A full prediction pipeline was saved using `joblib`.

Example usage:

```bash
python src/predict.py
```

Output:

```
Prediction: Churn
Probability: 0.918
```

This simulates real-world inference.

---

## 📁 11. Project Structure

```
Customer-Churn-Prediction/
│
├── data/
├── notebooks/
├── src/
│   ├── train_model.py
│   └── predict.py
├── models/
│   └── churn_pipeline.pkl
├── requirements.txt
└── README.md
```

---

## 🛠 12. Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Joblib

---

## 🎯 13. Future Improvements

* Deploy using Streamlit web app
* Use XGBoost / LightGBM
* Add SHAP explainability
* Hyperparameter tuning

---

## 👨‍💻 14. Resume Bullet Point

> Built a customer churn prediction system using Logistic Regression and Random Forest with full preprocessing pipeline, achieving ROC-AUC of 0.84 and recall of 0.79. Extracted key business insights and created deployable prediction script.

---

## ⭐ 15. How to Run

```bash
pip install -r requirements.txt
python src/train_model.py
python src/predict.py
```



