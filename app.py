import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight-Risk Radar",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(" Flight-Risk Radar")

st.markdown(
    """
    ### People Analytics Dashboard

    This application uses employee information and a machine
    learning model to estimate **flight-risk (attrition risk)**.

    The purpose is to help HR identify employees who may benefit
    from supportive conversations and appropriate interventions.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("Employee-Attrition.csv")


df_original = load_data()


# ============================================================
# PREPARE DATA
# ============================================================

df = df_original.copy()

df["Attrition"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# Remove columns that should not be used by the model
columns_to_drop = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber"
]

df_model = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# Separate X and y
X = df_model.drop(
    columns=["Attrition"]
)

y = df_model["Attrition"]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X_train, y_train):

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                SVC(
                    class_weight="balanced",
                    probability=True,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


trained_model = train_model(
    X_train,
    y_train
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_risk(score):

    if score >= 70:
        return "High"

    elif score >= 40:
        return "Medium"

    return "Low"


def get_risk_icon(level):

    if level == "High":
        return "🔴"

    elif level == "Medium":
        return "🟠"

    return "🟢"


def get_attention_signals(employee):

    signals = []

    if employee.get("OverTime") == "Yes":
        signals.append(
            "The employee currently works overtime."
        )

    if employee.get("JobSatisfaction", 5) <= 2:
        signals.append(
            "Job satisfaction is relatively low."
        )

    if employee.get("EnvironmentSatisfaction", 5) <= 2:
        signals.append(
            "Environment satisfaction is relatively low."
        )

    if employee.get("WorkLifeBalance", 5) <= 2:
        signals.append(
            "Work-life balance is relatively low."
        )

    if employee.get("YearsAtCompany", 100) <= 2:
        signals.append(
            "The employee has relatively short tenure at the company."
        )

    if employee.get("YearsSinceLastPromotion", 0) >= 5:
        signals.append(
            "A relatively long period has passed since the last promotion."
        )

    return signals


# ============================================================
# EXISTING EMPLOYEE PREDICTIONS
# ============================================================

risk_probability = trained_model.predict_proba(
    X
)[:, 1]


risk_df = df_original.copy()

risk_df["Risk Score"] = (
    risk_probability * 100
)

risk_df["Risk Level"] = (
    risk_df["Risk Score"]
    .apply(classify_risk)
)

risk_df = risk_df.sort_values(
    by="Risk Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(" Dashboard Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        " Dashboard",
        " Predict New Employee",
        " Batch Prediction",
        " Employee Details"
    ]
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == " Dashboard":

    st.header(" Risk Overview")

    total_employees = len(risk_df)

    high_risk = (
        risk_df["Risk Level"] == "High"
    ).sum()

    medium_risk = (
        risk_df["Risk Level"] == "Medium"
    ).sum()

    low_risk = (
        risk_df["Risk Level"] == "Low"
    ).sum()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Employees",
            total_employees
        )

    with col2:
        st.metric(
            "🔴 High Risk",
            high_risk
        )

    with col3:
        st.metric(
            "🟠 Medium Risk",
            medium_risk
        )

    with col4:
        st.metric(
            "🟢 Low Risk",
            low_risk
        )


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(" Risk Distribution")

    col1, col2 = st.columns(2)


    with col1:

        risk_counts = (
            risk_df["Risk Level"]
            .value_counts()
            .reindex(
                [
                    "High",
                    "Medium",
                    "Low"
                ]
            )
            .fillna(0)
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Number of Employees"
        ]

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Number of Employees",
            title="Employees by Risk Level",
            text="Number of Employees"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        fig = px.histogram(
            risk_df,
            x="Risk Score",
            nbins=20,
            title="Flight-Risk Score Distribution"
        )

        fig.update_layout(
            xaxis_title="Predicted Flight-Risk Score (%)",
            yaxis_title="Number of Employees"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # RISK FILTER
    # --------------------------------------------------------

    st.subheader(" Employee Flight-Risk Radar")

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        [
            "All",
            "High",
            "Medium",
            "Low"
        ]
    )


    if risk_filter == "All":

        filtered_df = risk_df.copy()

    else:

        filtered_df = risk_df[
            risk_df["Risk Level"] == risk_filter
        ].copy()


    display_columns = [
        "EmployeeNumber",
        "Age",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "JobSatisfaction",
        "OverTime",
        "YearsAtCompany",
        "Risk Score",
        "Risk Level"
    ]


    available_columns = [
        c for c in display_columns
        if c in filtered_df.columns
    ]


    display_df = filtered_df[
        available_columns
    ].copy()


    display_df["Risk Score"] = (
        display_df["Risk Score"].round(1)
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # TOP HIGH-RISK EMPLOYEES
    # --------------------------------------------------------

    st.subheader(
        " Top Employees Requiring Attention"
    )


    top_n = st.slider(
        "Number of employees",
        min_value=5,
        max_value=20,
        value=10
    )


    top_risk = risk_df.head(
        top_n
    ).copy()


    top_columns = [
        "EmployeeNumber",
        "Age",
        "Department",
        "JobRole",
        "OverTime",
        "JobSatisfaction",
        "WorkLifeBalance",
        "YearsAtCompany",
        "Risk Score",
        "Risk Level"
    ]


    top_columns = [
        c for c in top_columns
        if c in top_risk.columns
    ]


    top_risk["Risk Score"] = (
        top_risk["Risk Score"].round(1)
    )


    st.dataframe(
        top_risk[top_columns],
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # THRESHOLD EXPLANATION
    # --------------------------------------------------------

    with st.expander(
        "ℹ️ How are risk levels determined?"
    ):

        st.write(
            """
            The machine learning model estimates the probability
            that an employee belongs to the attrition class.

            The probability is converted into a percentage-based
            Risk Score.

            **Risk bands:**

            🔴 High Risk — 70% or above

            🟠 Medium Risk — 40% to below 70%

            🟢 Low Risk — below 40%

            These are prioritization bands. A high-risk prediction
            does not mean that an employee is guaranteed to leave.
            """
        )


# ============================================================
# PAGE 2 — PREDICT NEW EMPLOYEE
# ============================================================

elif page == " Predict New Employee":

    st.header(" Predict Flight Risk for a New Employee")

    st.write(
        """
        Enter the employee's information below and click
        **Predict Flight Risk**.
        """
    )


    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader(" Personal Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            sorted(
                X["Gender"].unique()
            )
        )

        marital_status = st.selectbox(
            "Marital Status",
            sorted(
                X["MaritalStatus"].unique()
            )
        )


    with col2:

        business_travel = st.selectbox(
            "Business Travel",
            sorted(
                X["BusinessTravel"].unique()
            )
        )

        distance_from_home = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=100,
            value=10
        )

        education = st.selectbox(
            "Education Level",
            sorted(
                X["Education"].unique()
            )
        )


    with col3:

        education_field = st.selectbox(
            "Education Field",
            sorted(
                X["EducationField"].unique()
            )
        )

        num_companies = st.number_input(
            "Number of Companies Worked",
            min_value=0,
            max_value=20,
            value=2
        )

        total_working_years = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=50,
            value=5
        )


    # --------------------------------------------------------
    # JOB INFORMATION
    # --------------------------------------------------------

    st.subheader(" Job Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        department = st.selectbox(
            "Department",
            sorted(
                X["Department"].unique()
            )
        )

        job_role = st.selectbox(
            "Job Role",
            sorted(
                X["JobRole"].unique()
            )
        )

        job_level = st.selectbox(
            "Job Level",
            sorted(
                X["JobLevel"].unique()
            )
        )


    with col2:

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=100,
            max_value=100000,
            value=5000
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            min_value=100,
            max_value=50000,
            value=15000
        )

        daily_rate = st.number_input(
            "Daily Rate",
            min_value=100,
            max_value=2000,
            value=800
        )


    with col3:

        hourly_rate = st.number_input(
            "Hourly Rate",
            min_value=1,
            max_value=200,
            value=65
        )

        percent_salary_hike = st.number_input(
            "Percent Salary Hike",
            min_value=0,
            max_value=50,
            value=15
        )

        performance_rating = st.selectbox(
            "Performance Rating",
            sorted(
                X["PerformanceRating"].unique()
            )
        )


    # --------------------------------------------------------
    # SATISFACTION AND WORK CONDITIONS
    # --------------------------------------------------------

    st.subheader(
        " Satisfaction & Work Conditions"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            sorted(
                X["EnvironmentSatisfaction"].unique()
            )
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            sorted(
                X["JobSatisfaction"].unique()
            )
        )

        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction",
            sorted(
                X["RelationshipSatisfaction"].unique()
            )
        )


    with col2:

        job_involvement = st.selectbox(
            "Job Involvement",
            sorted(
                X["JobInvolvement"].unique()
            )
        )

        work_life_balance = st.selectbox(
            "Work-Life Balance",
            sorted(
                X["WorkLifeBalance"].unique()
            )
        )

        overtime = st.selectbox(
            "OverTime",
            sorted(
                X["OverTime"].unique()
            )
        )


    with col3:

        stock_option = st.selectbox(
            "Stock Option Level",
            sorted(
                X["StockOptionLevel"].unique()
            )
        )

        training_times = st.number_input(
            "Training Times Last Year",
            min_value=0,
            max_value=20,
            value=3
        )


    # --------------------------------------------------------
    # TENURE
    # --------------------------------------------------------

    st.subheader(" Career & Tenure")

    col1, col2, col3 = st.columns(3)


    with col1:

        years_company = st.number_input(
            "Years At Company",
            min_value=0,
            max_value=50,
            value=5
        )


    with col2:

        years_current_role = st.number_input(
            "Years In Current Role",
            min_value=0,
            max_value=30,
            value=2
        )


    with col3:

        years_promotion = st.number_input(
            "Years Since Last Promotion",
            min_value=0,
            max_value=30,
            value=1
        )


    years_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=30,
        value=2
    )


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    st.markdown("---")

    predict_button = st.button(
        " Predict Flight Risk",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        # Build new employee record

        new_employee = pd.DataFrame(
            [{
                "Age": age,
                "BusinessTravel": business_travel,
                "DailyRate": daily_rate,
                "Department": department,
                "DistanceFromHome": distance_from_home,
                "Education": education,
                "EducationField": education_field,
                "EnvironmentSatisfaction": environment_satisfaction,
                "Gender": gender,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobRole": job_role,
                "JobSatisfaction": job_satisfaction,
                "MaritalStatus": marital_status,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies,
                "OverTime": overtime,
                "PercentSalaryHike": percent_salary_hike,
                "PerformanceRating": performance_rating,
                "RelationshipSatisfaction": relationship_satisfaction,
                "StockOptionLevel": stock_option,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_company,
                "YearsInCurrentRole": years_current_role,
                "YearsSinceLastPromotion": years_promotion,
                "YearsWithCurrManager": years_manager
            }]
        )


        # Make prediction

        probability = trained_model.predict_proba(
            new_employee
        )[0][1]


        risk_score = probability * 100

        risk_level = classify_risk(
            risk_score
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            " Prediction Result"
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Flight-Risk Score",
                f"{risk_score:.1f}%"
            )


        with result_col2:

            st.metric(
                "Risk Level",
                f"{get_risk_icon(risk_level)} {risk_level}"
            )


        # Large result message

        if risk_level == "High":

            st.error(
                f"""
                🔴 **High Flight Risk**

                The model estimates a relatively high probability
                of attrition for this employee: **{risk_score:.1f}%**.
                """
            )

        elif risk_level == "Medium":

            st.warning(
                f"""
                🟠 **Medium Flight Risk**

                The model estimates a moderate probability
                of attrition: **{risk_score:.1f}%**.
                """
            )

        else:

            st.success(
                f"""
                🟢 **Low Flight Risk**

                The model estimates a relatively low probability
                of attrition: **{risk_score:.1f}%**.
                """
            )


        # ----------------------------------------------------
        # HR ATTENTION SIGNALS
        # ----------------------------------------------------

        st.subheader(
            " HR Attention Signals"
        )


        employee_dict = new_employee.iloc[0].to_dict()

        signals = get_attention_signals(
            employee_dict
        )


        if signals:

            for signal in signals:

                st.warning(
                    f"• {signal}"
                )

        else:

            st.info(
                """
                No simple attention signal was identified
                from the selected profile fields.
                """
            )


        # ----------------------------------------------------
        # PROFILE
        # ----------------------------------------------------

        with st.expander(
            " View Entered Employee Information"
        ):

            profile = pd.DataFrame({
                "Feature": new_employee.columns,
                "Value": new_employee.iloc[0].values
            })

            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# PAGE 3 — BATCH PREDICTION
# ============================================================

elif page == " Batch Prediction":

    st.header(
        " Batch Employee Flight-Risk Prediction"
    )

    st.write(
        """
        Upload a CSV containing employee information. The model
        will calculate a flight-risk score for every employee.
        """
    )


    uploaded_file = st.file_uploader(
        "Upload employee CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            upload_df = pd.read_csv(
                uploaded_file
            )


            st.success(
                f"Successfully loaded {len(upload_df)} employee records."
            )


            st.subheader(
                " Uploaded Data Preview"
            )


            st.dataframe(
                upload_df.head(10),
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # CHECK REQUIRED FEATURES
            # ------------------------------------------------

            required_features = X.columns.tolist()


            missing_columns = [
                column
                for column in required_features
                if column not in upload_df.columns
            ]


            if missing_columns:

                st.error(
                    "The uploaded CSV is missing required columns:"
                )

                st.write(
                    missing_columns
                )


                st.info(
                    """
                    Please make sure your CSV contains the same
                    employee feature columns used by the model.
                    """
                )


            else:

                # Keep only model features
                prediction_input = upload_df[
                    required_features
                ].copy()


                # Predict
                probabilities = trained_model.predict_proba(
                    prediction_input
                )[:, 1]


                upload_df["Risk Score"] = (
                    probabilities * 100
                )


                upload_df["Risk Level"] = (
                    upload_df["Risk Score"]
                    .apply(classify_risk)
                )


                upload_df["Risk Score"] = (
                    upload_df["Risk Score"].round(1)
                )


                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                st.subheader(
                    " Batch Prediction Summary"
                )


                batch_high = (
                    upload_df["Risk Level"] == "High"
                ).sum()


                batch_medium = (
                    upload_df["Risk Level"] == "Medium"
                ).sum()


                batch_low = (
                    upload_df["Risk Level"] == "Low"
                ).sum()


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "Employees",
                        len(upload_df)
                    )


                with col2:

                    st.metric(
                        "🔴 High",
                        batch_high
                    )


                with col3:

                    st.metric(
                        "🟠 Medium",
                        batch_medium
                    )


                with col4:

                    st.metric(
                        "🟢 Low",
                        batch_low
                    )


                # ------------------------------------------------
                # RESULTS
                # ------------------------------------------------

                st.subheader(
                    " Prediction Results"
                )


                st.dataframe(
                    upload_df,
                    use_container_width=True,
                    hide_index=True
                )


                # ------------------------------------------------
                # HIGH RISK EMPLOYEES
                # ------------------------------------------------

                st.subheader(
                    "🚨 High-Risk Employees"
                )


                high_risk_df = upload_df[
                    upload_df["Risk Level"] == "High"
                ].copy()


                if high_risk_df.empty:

                    st.success(
                        "No high-risk employees were identified."
                    )

                else:

                    st.dataframe(
                        high_risk_df,
                        use_container_width=True,
                        hide_index=True
                    )


                # ------------------------------------------------
                # DOWNLOAD
                # ------------------------------------------------

                csv_data = upload_df.to_csv(
                    index=False
                ).encode("utf-8")


                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name="flight_risk_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                f"Error processing the uploaded file: {e}"
            )


# ============================================================
# PAGE 4 — EMPLOYEE DETAILS
# ============================================================

elif page == " Employee Details":

    st.header(
        " Individual Employee Risk Details"
    )


    employee_numbers = (
        risk_df["EmployeeNumber"]
        .tolist()
    )


    selected_employee = st.selectbox(
        "Select Employee Number",
        employee_numbers
    )


    employee = risk_df[
        risk_df["EmployeeNumber"]
        == selected_employee
    ].iloc[0]


    risk_score = float(
        employee["Risk Score"]
    )


    risk_level = employee[
        "Risk Level"
    ]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Risk Score",
            f"{risk_score:.1f}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            f"{get_risk_icon(risk_level)} {risk_level}"
        )


    with col3:

        actual_attrition = employee.get(
            "Attrition",
            "N/A"
        )

        st.metric(
            "Actual Attrition",
            actual_attrition
        )


    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    st.subheader(
        "Employee Profile"
    )


    profile_columns = [
        "EmployeeNumber",
        "Age",
        "Gender",
        "Department",
        "JobRole",
        "MaritalStatus",
        "BusinessTravel",
        "MonthlyIncome",
        "JobLevel",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "WorkLifeBalance",
        "OverTime",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "NumCompaniesWorked"
    ]


    profile_columns = [
        c for c in profile_columns
        if c in employee.index
    ]


    profile = pd.DataFrame({
        "Feature": profile_columns,
        "Value": [
            employee[c]
            for c in profile_columns
        ]
    })


    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # ATTENTION SIGNALS
    # --------------------------------------------------------

    st.subheader(
        " HR Attention Signals"
    )


    signals = get_attention_signals(
        employee.to_dict()
    )


    if signals:

        for signal in signals:

            st.warning(
                f"• {signal}"
            )

    else:

        st.info(
            "No simple attention signal was identified."
        )


# ============================================================
# ETHICS
# ============================================================

st.markdown("---")

with st.expander(
    "⚠️ Responsible & Ethical Use"
):

    st.markdown(
        """
        **Important:**

        The flight-risk score is a machine-learning prediction,
        not a fact about an employee.

        A high score does **not** mean that an employee will
        definitely leave the organization.

        The dashboard should be used to support:

        - constructive HR conversations
        - employee support
        - retention planning
        - workforce analysis

        It should **not** be used as the sole basis for:

        - termination
        - disciplinary action
        - promotion decisions
        - salary decisions
        - denying opportunities

        HR should consider the employee's context and use the
        model as a decision-support tool rather than an automatic
        decision-maker.

        The simple HR attention signals shown in this dashboard
        are descriptive indicators. They should not be interpreted
        as causal explanations for employee attrition.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
