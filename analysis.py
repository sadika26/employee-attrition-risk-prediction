import pandas as pd

df = pd.read_csv("Palo Alto Networks.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df["Attrition"].value_counts())
print(df.isnull().sum())
print(df["Attrition"].value_counts())

print(df["Department"].value_counts())
print(df.groupby("Department")["Attrition"].mean())
import matplotlib.pyplot as plt

df.groupby("Department")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Department")
plt.ylabel("Attrition Rate")
plt.xlabel("Department")
plt.show()
print(df.groupby("Age")["Attrition"].mean())
df.groupby("Age")["Attrition"].mean().plot(kind="line")

plt.title("Attrition Rate by Age")
plt.ylabel("Attrition Rate")
plt.xlabel("Age")
plt.show()
print(df.groupby("Gender")["Attrition"].mean())
df.groupby("Gender")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Gender")
plt.ylabel("Attrition Rate")
plt.xlabel("Gender")
plt.show()
print(df.groupby("OverTime")["Attrition"].mean())
df.groupby("OverTime")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Overtime")
plt.ylabel("Attrition Rate")
plt.xlabel("Overtime")
plt.show()
print(df.groupby("JobSatisfaction")["Attrition"].mean())
df.groupby("JobSatisfaction")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Job Satisfaction")
plt.ylabel("Attrition Rate")
plt.xlabel("Job Satisfaction")
plt.show()
print(df.groupby("JobLevel")["Attrition"].mean())
df.groupby("JobLevel")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Job Level")
plt.ylabel("Attrition Rate")
plt.xlabel("Job Level")
plt.show()
print(df.groupby("BusinessTravel")["Attrition"].mean())
print(df.groupby("BusinessTravel")["Attrition"].mean())

df.groupby("BusinessTravel")["Attrition"].mean().plot(kind="bar")
print(df.groupby("YearsAtCompany")["Attrition"].mean())
df.groupby("YearsAtCompany")["Attrition"].mean().plot(kind="line")

plt.title("Attrition Rate by Years at Company")
plt.ylabel("Attrition Rate")
plt.xlabel("Years at Company")
plt.show()
print(df.groupby("JobRole")["Attrition"].mean())
df.groupby("JobRole")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Job Role")
plt.ylabel("Attrition Rate")
plt.xlabel("Job Role")
plt.xticks(rotation=45, ha="right")
plt.show()
print(df.groupby("DistanceFromHome")["Attrition"].mean())
df.groupby("DistanceFromHome")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Distance From Home")
plt.ylabel("Attrition Rate")
plt.xlabel("Distance From Home")
plt.show()
print(df.groupby("MonthlyIncome")["Attrition"].mean())
print(df.groupby("MonthlyIncome")["Attrition"].mean())
print(df.groupby("WorkLifeBalance")["Attrition"].mean())
df.groupby("WorkLifeBalance")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Work-Life Balance")
plt.ylabel("Attrition Rate")
plt.xlabel("Work-Life Balance")
plt.show()
print(df.groupby("EnvironmentSatisfaction")["Attrition"].mean())
df.groupby("EnvironmentSatisfaction")["Attrition"].mean().plot(kind="bar")

plt.title("Attrition Rate by Environment Satisfaction")
plt.ylabel("Attrition Rate")
plt.xlabel("Environment Satisfaction")
plt.show()
# Feature Engineering

# 1. Income-to-Experience Ratio
df["IncomePerWorkingYear"] = (
    df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
)

# 2. Promotion Delay Indicators
df["PromotionDelayRatio"] = (
    df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
)

df["PromotionDelayFlag"] = (
    df["YearsSinceLastPromotion"] >= 3
).astype(int)

# 3. Engagement Score
engagement_columns = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]

df["EngagementScore"] = df[engagement_columns].mean(axis=1)

# 4. Workload Stress Flag
df["WorkloadStressFlag"] = (
    (df["OverTime"] == "Yes") &
    (
        (df["WorkLifeBalance"] <= 2) |
        (df["JobInvolvement"] <= 2)
    )
).astype(int)

print("\nFeature Engineering Completed")
print("New Features Added:")
print([
    "IncomePerWorkingYear",
    "PromotionDelayRatio",
    "PromotionDelayFlag",
    "EngagementScore",
    "WorkloadStressFlag"
])
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.preprocessing import LabelEncoder

X_encoded = X.copy()

le = LabelEncoder()

for column in X_encoded.select_dtypes(include="object").columns:
    X_encoded[column] = le.fit_transform(X_encoded[column])

print(X_encoded.head())
print(X_encoded.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training data:", X_train_scaled.shape)
print("Scaled testing data:", X_test_scaled.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Predictions:", y_pred[:10])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Predictions:", rf_pred[:10])
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Predictions:", rf_pred[:10])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Precision:", precision_score(y_test, rf_pred))
print("Random Forest Recall:", recall_score(y_test, rf_pred))
print("Random Forest F1 Score:", f1_score(y_test, rf_pred))
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(random_state=42)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

print("Gradient Boosting Predictions:", gb_pred[:10])
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Gradient Boosting Precision:", precision_score(y_test, gb_pred))
print("Gradient Boosting Recall:", recall_score(y_test, gb_pred))
print("Gradient Boosting F1 Score:", f1_score(y_test, gb_pred))
from sklearn.metrics import roc_auc_score

lr_prob = model.predict_proba(X_test_scaled)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]
gb_prob = gb_model.predict_proba(X_test)[:, 1]

print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr_prob))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_prob))
print("Gradient Boosting ROC-AUC:", roc_auc_score(y_test, gb_prob))
risk_probability = model.predict_proba(X_test_scaled)[:, 1]
# Threshold Comparison

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\nThreshold Comparison:")

for threshold in thresholds:
    threshold_pred = (risk_probability >= threshold).astype(int)

    precision = precision_score(y_test, threshold_pred, zero_division=0)
    recall = recall_score(y_test, threshold_pred, zero_division=0)
    f1 = f1_score(y_test, threshold_pred, zero_division=0)

    print(
        f"Threshold: {threshold} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1 Score: {f1:.2f}"
    )

risk_score = risk_probability * 100

print("Employee Risk Scores:", risk_score[:10])
risk_category = []

for score in risk_score:
    if score < 30:
        risk_category.append("Low")
    elif score < 60:
        risk_category.append("Medium")
    else:
        risk_category.append("High")

print("Risk Categories:", risk_category[:10])
print("\nRisk Category Distribution:")
print(pd.Series(risk_category).value_counts())
import matplotlib.pyplot as plt

risk_counts = pd.Series(risk_category).value_counts()

risk_counts.plot(kind="bar")
plt.title("Employee Risk Category Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)

plt.savefig("risk_distribution.png")
plt.show()
# Feature Importance using Logistic Regression

feature_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": model.coef_[0]
})

feature_importance["Absolute Importance"] = (
    feature_importance["Importance"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Absolute Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
# Feature Importance Graph

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Absolute Importance"]
)

plt.title("Top 10 Important Features")
plt.xlabel("Absolute Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()

plt.savefig("feature_importance.png")
plt.show()
# Save Model and Preprocessing Objects

import joblib

joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X_encoded.columns), "feature_columns.pkl")

print("\nModel and preprocessing objects saved successfully!")