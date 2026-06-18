"""
Why Compare Ensemble Models
    Excels in different scenarios
    Helps identify the most effective model for a specific dataset or problem
Ensemble models to consider
    Bagging (Eg. Random Forest)
        Reduces variance by averaging predictions from multiple independent models
        Works well with high- variance models like decision trees
    Boosting (Eg. Gradient Boosting)
        Reduces bias by sequentially correcting errors from previous models
        Works well with low- bias models like decision trees
        Effective for complex datasets and imbalanced data
"""

import pickle

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data/telecom-churn.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df["Churn"].value_counts())

# Handle Missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna({"TotalCharges": df["TotalCharges"].median()}, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=["object"]).columns:
    if column != "Churn":
        df[column] = label_encoder.fit_transform(df[column])

# Encode target variable
df["Churn"] = label_encoder.fit_transform(df["Churn"])

# Scale the numerical features
scaler = StandardScaler()
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split the data into features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Display Class Distribution
print(pd.Series(y_train_resampled).value_counts())


# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_classifier.predict(X_test)

# Train XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=42, eval_metric="logloss")
xgb_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_classifier.predict(X_test)

# Train LightGBM Classifier
lgbm_classifier = LGBMClassifier(random_state=42)
lgbm_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_lgbm = lgbm_classifier.predict(X_test)


# Train CatBoost Classifier
cat_classifier = CatBoostClassifier(random_state=42, verbose=0)
cat_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_cat = cat_classifier.predict(X_test)

# Evaluate the models
#
print(f"""
    Classification Reports


    RF:
        {classification_report(y_test, y_pred_rf)}

    XGBoost:
        {classification_report(y_test, y_pred_xgb)}

    LightGBM:
        {classification_report(y_test, y_pred_lgbm)}

    CatBoost:
        {classification_report(y_test, y_pred_cat)}


    Random Forest: {roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])}
    XGBoost: {roc_auc_score(y_test, xgb_classifier.predict_proba(X_test)[:, 1])}
    LightGBM: {roc_auc_score(y_test, lgbm_classifier.predict_proba(X_test)[:, 1])}
    CatBoost: {roc_auc_score(y_test, cat_classifier.predict_proba(X_test)[:, 1])}
    """)


# Save the models
with open("models/week7/rf_classifier.pkl", "wb") as f:
    pickle.dump(rf_classifier, f)
with open("models/week7/xgb_classifier.pkl", "wb") as f:
    pickle.dump(xgb_classifier, f)
with open("models/week7/lgbm_classifier.pkl", "wb") as f:
    pickle.dump(lgbm_classifier, f)
with open("models/week7/cat_classifier.pkl", "wb") as f:
    pickle.dump(cat_classifier, f)
