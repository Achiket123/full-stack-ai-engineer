"""
LightGBM
Implementation of gradient boosting designed to handle large datasets and high -dimensional data with speed and accuracy
Key Features
 - Histogram-based Splitting
 - Leaf wise tree growth:
 - Support for GPU training
 - Handles sparse data
Advantages
 - Faster training times than XGBoost
 - Lower memory usage
 - Better accuracy
When to use
Large datasets with numerical features
Time sensitive tasks requiring fast predictions

CatBoost
Implementation of gradient boosting designed to handle categorical features efficiently
without the need for one-hot encoding
Key Features
 - Native support for categorical features
 - ordered boosting
 - robust overfitting
Advantages
 - Eliminate the need for one-hot encoding or manual encoding
 - reduces overfitting
 - easy to implement for datasets with many categorical features
When to use
Datasets with many categorical features
Applications where overfitting is a concern
"""

import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("data/titanic.csv")
# handle missing values
df.fillna(
    {"Age": df["Age"].median(), "Embarked": df["Embarked"].mode()[0]}, inplace=True
)

# Encode categorical features
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
target = df["Survived"]

# split
X = features
y = target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

catboost_model = CatBoostClassifier(random_state=42)
catboost_model.fit(X_train, y_train)

y_pred = catboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("CatBoost Accuracy:", accuracy)

xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)


# train catboost without encoding categorical features
cat_boost_native = CatBoostClassifier(
    cat_features=["Sex", "Embarked"],
    verbose=0,
    random_state=42,
)
cat_boost_native.fit(X_train, y_train)

y_pred = cat_boost_native.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("CatBoost Native Accuracy:", accuracy)
