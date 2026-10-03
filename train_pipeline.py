import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# 1. Load Dataset
df = pd.read_csv("Palo Alto Networks.csv")


# 2. Feature Engineering

df["IncomePerWorkingYear"] = (
    df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
)

df["PromotionDelayRatio"] = (
    df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
)

df["PromotionDelayFlag"] = (
    df["YearsSinceLastPromotion"] >= 3
).astype(int)

engagement_columns = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]

df["EngagementScore"] = df[engagement_columns].mean(axis=1)

df["WorkloadStressFlag"] = (
    (df["OverTime"] == "Yes") &
    (
        (df["WorkLifeBalance"] <= 2) |
        (df["JobInvolvement"] <= 2)
    )
).astype(int)


# 3. Separate Features and Target

X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# 4. Identify Column Types

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "str"]
).columns.tolist()


# 5. Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# 6. Logistic Regression Pipeline

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)


# 7. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 8. Train Model

pipeline.fit(X_train, y_train)


# 9. Predictions

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]


# 10. Model Evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)


print("\nDashboard Pipeline Model Results:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
print(f"ROC-AUC: {roc_auc:.2%}")


# 11. Save Complete Pipeline

joblib.dump(
    pipeline,
    "attrition_dashboard_pipeline.pkl"
)


print("\nDashboard pipeline saved successfully!")