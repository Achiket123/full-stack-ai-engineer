"""
OneHot Encoding and label encoding
What are categorical variables
    Binary Categorical features Gender(Male/Female)
    Multi-Class Categorical Features (Countries)

One-Hot Encoding
    Creates binary columns for each categorical feature
    Each row is marked with a 1 for its respective category adn 0 elesewhere
    Example Feature COlor = [RED,BLUE, GREEN]
    Color_RED, Color_BLUE, Color_GREEN
    1 0 0
    0 1 0
    0 0 1
Label Encoding
    Label Encoding assigns a unique integer to each category
    Example Feature Color = [RED, BLUE, GREEN]
    Color = [0, 1, 2]
Dealing With High Cardinality and Categorical Features
High-cardinality categorical features contain a large number of unique categories
Challenges
    Dimensionality
    Sparse Representaion
Solutions
    Frequency Encoding
        Replaces each category with its frequency in the dataset
        Example Feature Color = [RED, BLUE, GREEN]
        Color = [2, 1, 1]
    Target Encoding
        Replace categories witht the mean of the target variable for each category

When To Use Different Encoding techinques:
    One-Hot Encoding
        Use when the number of unique categories is small and the categories are not ordinal
    Label Encoding
        Assigns a unique integer to each category
    Frequency Encoding
        Replaces each category with its frequency in the dataset, in both regression and classification tasks
    Target Encoding
        Replaces categories with the mean of the target variable for each category
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/titanic.csv")
print(df.info())

print(df.head())

# Apply one-hot encoding
df_one_hot = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

print("One Hot Encoded")
print(df_one_hot.head())

# Apply Label Encoding
label_encoder = LabelEncoder()
df["Pclass_encoded"] = label_encoder.fit_transform(df["Pclass"])

print("Label Encoded")
print(df[["Pclass", "Pclass_encoded"]].head())

# Apply frequency encoding
df["Ticket_frequency"] = df["Ticket"].map(df["Ticket"].value_counts())

print("Frequency Encoded")
print(df[["Ticket", "Ticket_frequency"]].head())

X = df_one_hot.drop(
    columns=[
        "Survived",
        "Name",
        "Ticket",
        "Age",
        "Cabin",
    ]
)


y = df["Survived"]
print(df_one_hot.isnull().sum())
# Apply train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
