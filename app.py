import streamlit as st
import pandas as pd
import joblib

# Load dataset and trained pipeline
df = pd.read_csv("Palo Alto Networks.csv")
pipeline = joblib.load("attrition_dashboard_pipeline.pkl")

# Page settings
st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Employee Attrition Risk Dashboard")

st.write(
    "Machine Learning-Based Employee Attrition Prediction "
    "and Risk Scoring System"
)

st.divider()

# Basic dataset information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Employees", len(df))

with col2:
    st.metric(
        "Employees Who Left",
        int((df["Attrition"] == 1).sum())
    )

with col3:
    st.metric(
        "Current Attrition Rate",
        f"{(df['Attrition'].mean() * 100):.2f}%"
    )

st.divider()

st.subheader("Employee Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)
# -----------------------------
# Individual Employee Risk
# -----------------------------

def add_features(data):
    data = data.copy()

    data["IncomePerWorkingYear"] = (
        data["MonthlyIncome"] / (data["TotalWorkingYears"] + 1)
    )

    data["PromotionDelayRatio"] = (
        data["YearsSinceLastPromotion"] /
        (data["YearsAtCompany"] + 1)
    )

    data["PromotionDelayFlag"] = (
        data["YearsSinceLastPromotion"] >= 3
    ).astype(int)

    engagement_columns = [
        "JobInvolvement",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance"
    ]

    data["EngagementScore"] = data[engagement_columns].mean(axis=1)

    data["WorkloadStressFlag"] = (
        (data["OverTime"] == "Yes") &
        (
            (data["WorkLifeBalance"] <= 2) |
            (data["JobInvolvement"] <= 2)
        )
    ).astype(int)

    return data


st.divider()

st.subheader("Individual Employee Risk Prediction")

selected_employee = st.selectbox(
    "Select Employee",
    range(len(df)),
    format_func=lambda x: f"Employee {x + 1}"
)

employee = df.iloc[[selected_employee]].copy()

employee_input = employee.drop(
    columns=["Attrition"],
    errors="ignore"
)

employee_input = add_features(employee_input)

risk_probability = pipeline.predict_proba(
    employee_input
)[0, 1]

risk_score = risk_probability * 100

if risk_score < 30:
    risk_category = "Low Risk"
elif risk_score < 60:
    risk_category = "Medium Risk"
else:
    risk_category = "High Risk"

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Attrition Risk Score",
        f"{risk_score:.2f}%"
    )

with col2:
    st.metric(
        "Risk Category",
        risk_category
    )
# -----------------------------
# Department-wise Risk Analysis
# -----------------------------

st.divider()

st.subheader("Department-wise Risk Analysis")

department_risk = (
    df.groupby("Department")["Attrition"]
    .mean()
    .reset_index()
)

department_risk["Attrition Risk (%)"] = (
    department_risk["Attrition"] * 100
)

st.dataframe(
    department_risk[
        ["Department", "Attrition Risk (%)"]
    ].round(2),
    width="stretch"
)
st.bar_chart(
    department_risk.set_index("Department")[
        "Attrition Risk (%)"
    ]
) 
# -----------------------------
# Risk Distribution
# -----------------------------

st.divider()

st.subheader("Overall Risk Distribution")

all_employees = df.copy()

all_employee_input = all_employees.drop(
    columns=["Attrition"],
    errors="ignore"
)

all_employee_input = add_features(all_employee_input)

all_risk_probability = pipeline.predict_proba(
    all_employee_input
)[:, 1]

all_risk_score = all_risk_probability * 100

# Risk Threshold Controls

st.subheader("Risk Threshold Settings")

low_threshold = st.slider(
    "Low Risk Upper Limit (%)",
    10,
    50,
    30
)

high_threshold = st.slider(
    "High Risk Lower Limit (%)",
    51,
    90,
    60
)

high_risk_count = int(
    (all_risk_score >= high_threshold).sum()
)

st.metric(
    "High-Risk Employees",
    high_risk_count
)

risk_categories = pd.cut(
    all_risk_score,
    bins=[-1, low_threshold, high_threshold, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

risk_distribution = risk_categories.value_counts().reindex(
    ["Low Risk", "Medium Risk", "High Risk"]
)
st.dataframe(
    risk_distribution.reset_index(
        name="Employee Count"
    ),
    width="stretch"
)

st.bar_chart(risk_distribution)
 # -----------------------------
# Dashboard Filters
# -----------------------------

st.divider()

st.subheader("Dashboard Filters")

col1, col2 = st.columns(2)

with col1:
    selected_department = st.selectbox(
        "Select Department",
        ["All"] + sorted(df["Department"].unique().tolist())
    )

with col2:
    selected_risk = st.selectbox(
        "Select Risk Category",
        ["All", "Low Risk", "Medium Risk", "High Risk"]
    )

filtered_df = df.copy()

# Department filter
if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["Department"] == selected_department
    ]
# Calculate risk for filtered employees
filtered_input = filtered_df.drop(
    columns=["Attrition"],
    errors="ignore"
)

filtered_input = add_features(filtered_input)

filtered_probability = pipeline.predict_proba(
    filtered_input
)[:, 1]

filtered_score = filtered_probability * 100

filtered_categories = pd.cut(
    filtered_score,
    bins=[-1, low_threshold, high_threshold, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Risk filter
if selected_risk != "All":
    filtered_df = filtered_df[
        filtered_categories == selected_risk
    ]

st.write(
    f"Showing **{len(filtered_df)} employees**"
)

st.dataframe(
    filtered_df,
    width="stretch"
)
# -----------------------------
# Job-role-wise Risk Analysis
# -----------------------------

st.divider()

st.subheader("Job-role-wise Risk Analysis")

job_role_risk = (
    df.groupby("JobRole")["Attrition"]
    .mean()
    .reset_index()
)

job_role_risk["Attrition Risk (%)"] = (
    job_role_risk["Attrition"] * 100
)

job_role_risk = job_role_risk.sort_values(
    "Attrition Risk (%)",
    ascending=False
)

st.dataframe(
    job_role_risk[
        ["JobRole", "Attrition Risk (%)"]
    ].round(2),
    width="stretch"
)

st.bar_chart(
    job_role_risk.set_index("JobRole")[
        "Attrition Risk (%)"
    ]
)
# -----------------------------
# Feature Importance
# -----------------------------

st.divider()

st.subheader("Top Factors Influencing Attrition Risk")

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": abs(model.coef_[0])
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
).head(10)

st.dataframe(
    importance_df,
    width="stretch"
)

st.bar_chart(
    importance_df.set_index("Feature")["Importance"]
)
# -----------------------------
# What-if Exploration
# -----------------------------

st.divider()

st.subheader("What-if Risk Exploration")

st.write(
    "Adjust employee factors to explore how changes may affect attrition risk."
)

what_if_employee = df.iloc[[selected_employee]].copy()

col1, col2, col3 = st.columns(3)

with col1:
    what_if_overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"],
        index=0 if what_if_employee["OverTime"].iloc[0] == "No" else 1
    )

with col2:
    what_if_job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        int(what_if_employee["JobSatisfaction"].iloc[0])
    )

with col3:
    what_if_worklife = st.slider(
        "Work-Life Balance",
        1,
        4,
        int(what_if_employee["WorkLifeBalance"].iloc[0])
    )

what_if_employee["OverTime"] = what_if_overtime
what_if_employee["JobSatisfaction"] = what_if_job_satisfaction
what_if_employee["WorkLifeBalance"] = what_if_worklife

what_if_input = what_if_employee.drop(
    columns=["Attrition"],
    errors="ignore"
)

what_if_input = add_features(what_if_input)

what_if_probability = pipeline.predict_proba(
    what_if_input
)[0, 1]

what_if_score = what_if_probability * 100

if what_if_score < 30:
    what_if_category = "Low Risk"
elif what_if_score < 60:
    what_if_category = "Medium Risk"
else:
    what_if_category = "High Risk"

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "What-if Risk Score",
        f"{what_if_score:.2f}%"
    )

with col2:
    st.metric(
        "What-if Risk Category",
        what_if_category
    )