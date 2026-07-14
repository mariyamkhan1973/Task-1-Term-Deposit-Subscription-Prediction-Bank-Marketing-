# Term Deposit Subscription Prediction — Bank Marketing

Predicting whether a bank customer will subscribe to a term deposit as a result of a
telemarketing campaign, using classification models and SHAP-based explainability.

##  Task Objective

Build a classification model that predicts term deposit subscription (`yes`/`no`) from
customer demographic and campaign-interaction data, evaluate it with metrics suited to an
imbalanced target, and explain individual predictions using SHAP.

## Approach

1. **EDA** — explored the distribution of the target, numeric features, and subscription
   rate by job category to understand the data before modeling.
2. **Preprocessing** — mapped binary yes/no columns to 0/1 and one-hot encoded nominal
   categorical features (`job`, `marital`, `education`, `contact`, `month`, `poutcome`).
3. **Modeling** — trained a `LogisticRegression` baseline and a `RandomForestClassifier`
   (both with `class_weight="balanced"` to handle the ~10% positive class).
4. **Evaluation** — compared models via Confusion Matrix, F1-score, and ROC-AUC/ROC curve.
5. **Explainability** — used `shap.TreeExplainer` on the Random Forest model to produce a
   global summary plot and to explain 5 individual customer predictions.

## Results & Findings

- The dataset is imbalanced (~10% subscribers), so F1-score and ROC-AUC were used instead
  of raw accuracy.
- Random Forest outperformed Logistic Regression on ROC-AUC by capturing non-linear
  feature interactions.
- SHAP identified **call duration**, **previous campaign outcome (`poutcome`)**, and
  **contact method** as the strongest drivers of subscription decisions.
- **Business takeaway:** prioritize follow-ups with customers who had a successful prior
  campaign outcome, prefer cellular contact, and coach agents toward longer, higher-quality
  conversations.

## Repository Structure



├── Term_Deposit_Prediction.ipynb   # Full notebook: EDA → modeling → SHAP explainability
├── data/
│   └── bank_marketing.csv          # Dataset (see note below)
├── requirements.txt
└── README.md



##  How to Run


pip install -r requirements.txt

jupyter notebook Term_Deposit_Prediction.ipynb

