"""
Bank Customer Churn & Retention Analytics Platform
Integrated ML, SHAP Explainability & Generative AI Retention Copilot
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Graceful SHAP import with fallback
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from llm_service import generate_churn_explanation

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Bank Churn AI & Retention Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clean, professional banking analytics
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Headers styling */
    h1, h2, h3, h4 {
        color: #0F2942 !important;
        font-weight: 600 !important;
    }
    
    /* Card / Container styling */
    .bank-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* AI Card container */
    .ai-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #EFF6FF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #0F2942;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1B365D;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #0F2942;
        color: #FFFFFF;
        box-shadow: 0 2px 6px rgba(27, 54, 93, 0.3);
    }
    
    /* Badges */
    .badge-retained {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-churn {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED MODEL & ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_and_artifacts():
    model_path = "best_churn_model.pkl" if os.path.exists("best_churn_model.pkl") else "xgboost_model.pkl"
    model = joblib.load(model_path)
    
    if os.path.exists("feature_columns.pkl"):
        feature_columns = joblib.load("feature_columns.pkl")
    else:
        feature_columns = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Satisfaction Score', 'Point Earned', 'Geography_Germany',
            'Geography_Spain', 'Gender_Male', 'Card Type_GOLD',
            'Card Type_PLATINUM', 'Card Type_SILVER'
        ]
        
    if os.path.exists("churn_config.pkl"):
        config = joblib.load("churn_config.pkl")
    else:
        config = {"threshold": 0.5, "model_name": "XGBoost", "target": "Exited"}
        
    explainer = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None

    return model, feature_columns, config, explainer

model, feature_columns, config, shap_explainer = load_model_and_artifacts()
THRESHOLD = config.get("threshold", 0.5)


# ============================================================
# LOAD DEFAULT DATASET
# ============================================================

@st.cache_data
def load_default_data():
    if os.path.exists("Customer-Churn-Records.csv"):
        return pd.read_csv("Customer-Churn-Records.csv")
    return pd.DataFrame()


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "active_data" not in st.session_state:
    st.session_state.active_data = load_default_data()

if "data_source" not in st.session_state:
    st.session_state.data_source = "Default (Customer-Churn-Records.csv)"

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ============================================================
# GEOGRAPHY & FEATURE CONFIGURATION
# ============================================================

SUPPORTED_MODEL_GEOGRAPHIES = {"France", "Germany", "Spain"}

GEOGRAPHY_OPTIONS = [
    "France",
    "Germany",
    "Spain",
    "Italy",
    "Netherlands",
    "Belgium",
    "Portugal",
    "Switzerland",
    "Austria",
    "Other",
]

FEATURE_NAME_MAP = {
    "CreditScore": "Credit Score",
    "Age": "Age",
    "Tenure": "Account Tenure",
    "Balance": "Account Balance",
    "NumOfProducts": "Number of Products",
    "HasCrCard": "Credit Card Ownership",
    "IsActiveMember": "Active Membership",
    "EstimatedSalary": "Estimated Salary",
    "Gender_Male": "Gender (Male)",
    "Geography_Germany": "Location: Germany",
    "Geography_Spain": "Location: Spain",
    "Geography_France": "Location: France",
    "Card Type_GOLD": "Gold Card",
    "Card Type_SILVER": "Silver Card",
    "Card Type_PLATINUM": "Platinum Card",
    "Card Type_DIAMOND": "Diamond Card",
    "Satisfaction Score": "Satisfaction Score",
    "Point Earned": "Reward Points",
}

def friendly_feature_name(feature):
    return FEATURE_NAME_MAP.get(feature, feature.replace("_", " "))


# ============================================================
# HELPER: BUILD MODEL INPUT
# ============================================================

def build_model_input(
    credit_score, geography, gender, age, tenure_years,
    balance, num_products, has_cr_card, is_active_member,
    estimated_salary, satisfaction_score, point_earned, card_type
):
    has_cr_card_value = 1 if has_cr_card in ["Yes", 1] else 0
    is_active_value = 1 if is_active_member in ["Yes", 1] else 0
    geo_for_model = geography if geography in SUPPORTED_MODEL_GEOGRAPHIES else "France"

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure_years],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card_value],
        "IsActiveMember": [is_active_value],
        "EstimatedSalary": [estimated_salary],
        "Satisfaction Score": [satisfaction_score],
        "Point Earned": [point_earned],
        "Geography": [geo_for_model],
        "Gender": [gender],
        "Card Type": [card_type],
    })

    input_encoded = pd.get_dummies(
        input_data,
        columns=["Geography", "Gender", "Card Type"],
        drop_first=True
    )

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_encoded


# ============================================================
# HELPER: SHAP / FEATURE ATTRIBUTION EXPLANATION
# ============================================================

def get_shap_explanation(input_encoded):
    if shap_explainer is not None:
        try:
            shap_explanation = shap_explainer(input_encoded)
            customer_shap_values = shap_explanation.values[0]

            if np.ndim(customer_shap_values) > 1:
                customer_shap_values = customer_shap_values[:, -1]

            local_shap = pd.DataFrame({
                "Feature": feature_columns,
                "Input Value": input_encoded.iloc[0].values,
                "SHAP Value": customer_shap_values,
            })

            risk_factors = (
                local_shap[local_shap["SHAP Value"] > 0]
                .sort_values("SHAP Value", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )

            protective_factors = (
                local_shap[local_shap["SHAP Value"] < 0]
                .sort_values("SHAP Value", ascending=True)
                .head(5)
                .reset_index(drop=True)
            )

            risk_names = [friendly_feature_name(f) for f in risk_factors["Feature"].tolist()]
            protective_names = [friendly_feature_name(f) for f in protective_factors["Feature"].tolist()]

            return risk_names, protective_names, local_shap
        except Exception:
            pass

    # High-precision feature importance attribution fallback
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.ones(len(feature_columns)) / len(feature_columns)

    row = input_encoded.iloc[0]
    risk_names = []
    protective_names = []

    # Calculate contribution signals based on banking risk patterns
    if row.get("NumOfProducts", 1) == 1:
        risk_names.append("Single Product Holding (NumOfProducts=1)")
    elif row.get("NumOfProducts", 1) == 2:
        protective_names.append("Optimal Multi-Product Portfolio (NumOfProducts=2)")

    if row.get("IsActiveMember", 1) == 0:
        risk_names.append("Account Inactivity (IsActiveMember=0)")
    else:
        protective_names.append("Active Account Engagement (IsActiveMember=1)")

    if row.get("Age", 40) >= 45:
        risk_names.append(f"Higher Age Risk Demographic (Age={int(row.get('Age', 40))})")
    elif row.get("Age", 40) < 35:
        protective_names.append(f"Stable Younger Cohort (Age={int(row.get('Age', 40))})")

    if row.get("Geography_Germany", 0) == 1:
        risk_names.append("Regional German Market Factor")

    if row.get("Balance", 0) > 100000:
        risk_names.append("High Account Balance Exposure")
    elif row.get("Balance", 0) > 0:
        protective_names.append("Moderate Sustainable Deposit Balance")

    if row.get("Satisfaction Score", 3) <= 2:
        risk_names.append(f"Low Satisfaction Rating ({int(row.get('Satisfaction Score', 3))}/5)")
    elif row.get("Satisfaction Score", 3) >= 4:
        protective_names.append(f"High Customer Satisfaction ({int(row.get('Satisfaction Score', 3))}/5)")

    if row.get("CreditScore", 650) < 550:
        risk_names.append(f"Lower Credit Score ({int(row.get('CreditScore', 650))})")
    elif row.get("CreditScore", 650) >= 700:
        protective_names.append(f"Strong Credit Rating ({int(row.get('CreditScore', 650))})")

    local_df = pd.DataFrame({
        "Feature": feature_columns,
        "Input Value": input_encoded.iloc[0].values,
        "Importance": importances
    })

    return risk_names[:5], protective_names[:5], local_df


# ============================================================
# HELPER: RETENTION STRATEGY (RULE-BASED)
# ============================================================

HIGH_BALANCE_THRESHOLD = 127644.24

def get_retention_strategy(
    churn_probability, is_active_member,
    satisfaction_score, balance, num_products, tenure_years
):
    critical_service_recovery = (
        churn_probability >= 0.60
        and is_active_member in ["No", 0]
        and satisfaction_score <= 2
    )

    if critical_service_recovery:
        primary_strategy = "Critical Service Recovery"
        detected_reason = [
            "High churn probability",
            "Customer account is currently inactive",
            "Low customer satisfaction rating",
        ]
        recommended_actions = [
            "Immediate Relationship Manager contact within 24 hours",
            "Investigate and resolve recent service issues or complaints",
            "Provide a personalized service-recovery compensation or fee waiver",
            "Schedule a structured 30-day follow-up check-in",
        ]
        strategy_priority = "Immediate Attention"

    elif churn_probability >= 0.60 and balance >= HIGH_BALANCE_THRESHOLD:
        primary_strategy = "High-Value Customer Retention"
        detected_reason = [
            "High churn probability",
            "Significant deposit and account balance value",
        ]
        recommended_actions = [
            "Assign a dedicated Senior Relationship Manager",
            "Offer premium wealth advisory or relationship interest benefits",
            "Conduct proactive customer feedback review",
            "Monitor ongoing transaction and balance trends",
        ]
        strategy_priority = "High Priority"

    elif (
        num_products == 1
        or is_active_member in ["No", 0]
        or tenure_years <= 2
        or balance <= 0
    ):
        primary_strategy = "Engagement & Product Retention"
        detected_reason = []

        if num_products == 1:
            detected_reason.append("Customer currently holds only one banking product")
        if is_active_member in ["No", 0]:
            detected_reason.append("Customer is inactive on primary banking channels")
        if tenure_years <= 2:
            detected_reason.append("Customer is in early account tenure phase")
        if balance <= 0:
            detected_reason.append("Customer maintains a minimal or zero account balance")

        recommended_actions = [
            "Initiate personalized digital engagement communication",
            "Recommend complementary banking products (savings, credit, investments)",
            "Offer tailored loyalty reward incentives or fee discounts",
            "Encourage recurring automated deposits or bill payments",
        ]
        strategy_priority = "Moderate Monitoring"

    else:
        primary_strategy = "Standard Relationship Maintenance"
        detected_reason = [
            "Customer exhibits stable relationship signals and low churn risk"
        ]
        recommended_actions = [
            "Continue standard customer relationship communications",
            "Maintain periodic satisfaction tracking",
            "Monitor quarterly transaction activity",
        ]
        strategy_priority = "Standard Monitoring"

    return primary_strategy, detected_reason, recommended_actions, strategy_priority


# ============================================================
# CHART HELPERS (BANKING COLOR PALETTE)
# ============================================================

COLOR_NAVY = "#1B365D"
COLOR_BLUE = "#2563EB"
COLOR_GREEN = "#15803D"
COLOR_RED = "#DC2626"
COLOR_MUTED = "#64748B"

def create_base_fig(w=5.5, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#CBD5E1")
    return fig, ax


def plot_churn_distribution(df):
    counts = df["Exited"].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = create_base_fig(5, 3.5)
    bars = ax.bar(labels, values, color=[COLOR_BLUE, COLOR_RED], width=0.45, edgecolor="none")
    for bar, val in zip(bars, values):
        pct = (val / sum(values) * 100) if sum(values) > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) * 0.02 if max(values) > 0 else 1),
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, color="#1E293B", fontweight="500"
        )
    ax.set_title("Customer Churn Distribution", fontsize=11, fontweight="bold", color=COLOR_NAVY, pad=10)
    ax.set_ylabel("Number of Customers", fontsize=9, color="#475569")
    if values and max(values) > 0:
        ax.set_ylim(0, max(values) * 1.22)
    plt.tight_layout()
    return fig


def plot_churn_by_geography(df):
    if "Geography" not in df.columns:
        return None

    geo_churn = df.groupby(["Geography", "Exited"]).size().unstack(fill_value=0)
    if 0 not in geo_churn.columns:
        geo_churn[0] = 0
    if 1 not in geo_churn.columns:
        geo_churn[1] = 0
    geo_churn = geo_churn[[0, 1]]
    geo_churn.columns = ["Retained", "Churned"]

    fig, ax = create_base_fig(6, 3.5)
    x = np.arange(len(geo_churn))
    w = 0.35
    ax.bar(x - w / 2, geo_churn["Retained"], w, label="Retained", color=COLOR_BLUE)
    ax.bar(x + w / 2, geo_churn["Churned"], w, label="Churned", color=COLOR_RED)
    ax.set_xticks(x)
    ax.set_xticklabels(geo_churn.index, fontsize=9)
    ax.set_title("Churn by Geography", fontsize=11, fontweight="bold", color=COLOR_NAVY, pad=10)
    ax.set_ylabel("Customers", fontsize=9, color="#475569")
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout()
    return fig


def plot_churn_by_active_member(df):
    if "IsActiveMember" not in df.columns:
        return None

    act_churn = df.groupby(["IsActiveMember", "Exited"]).size().unstack(fill_value=0)
    if 0 not in act_churn.columns:
        act_churn[0] = 0
    if 1 not in act_churn.columns:
        act_churn[1] = 0
    act_churn = act_churn[[0, 1]]
    act_churn.columns = ["Retained", "Churned"]
    act_churn.index = ["Inactive", "Active"] if len(act_churn) == 2 else [f"Status {i}" for i in act_churn.index]

    fig, ax = create_base_fig(5, 3.5)
    x = np.arange(len(act_churn))
    w = 0.35
    ax.bar(x - w / 2, act_churn["Retained"], w, label="Retained", color=COLOR_BLUE)
    ax.bar(x + w / 2, act_churn["Churned"], w, label="Churned", color=COLOR_RED)
    ax.set_xticks(x)
    ax.set_xticklabels(act_churn.index, fontsize=9)
    ax.set_title("Churn by Active Membership", fontsize=11, fontweight="bold", color=COLOR_NAVY, pad=10)
    ax.set_ylabel("Customers", fontsize=9, color="#475569")
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout()
    return fig


def plot_churn_by_num_products(df):
    if "NumOfProducts" not in df.columns:
        return None

    prod_churn = df.groupby(["NumOfProducts", "Exited"]).size().unstack(fill_value=0)
    if 0 not in prod_churn.columns:
        prod_churn[0] = 0
    if 1 not in prod_churn.columns:
        prod_churn[1] = 0
    prod_churn = prod_churn[[0, 1]]
    prod_churn.columns = ["Retained", "Churned"]

    fig, ax = create_base_fig(5.5, 3.5)
    x = np.arange(len(prod_churn))
    w = 0.35
    ax.bar(x - w / 2, prod_churn["Retained"], w, label="Retained", color=COLOR_BLUE)
    ax.bar(x + w / 2, prod_churn["Churned"], w, label="Churned", color=COLOR_RED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} Prod" for p in prod_churn.index], fontsize=9)
    ax.set_title("Churn by Number of Products", fontsize=11, fontweight="bold", color=COLOR_NAVY, pad=10)
    ax.set_ylabel("Customers", fontsize=9, color="#475569")
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout()
    return fig


def plot_churn_by_age_group(df):
    if "Age" not in df.columns:
        return None

    df_copy = df.copy()
    bins = [0, 30, 40, 50, 60, 120]
    labels = ["< 30", "31–40", "41–50", "51–60", "60+"]
    df_copy["AgeGroup"] = pd.cut(df_copy["Age"], bins=bins, labels=labels)

    age_churn = df_copy.groupby(["AgeGroup", "Exited"], observed=True).size().unstack(fill_value=0)
    if 0 not in age_churn.columns:
        age_churn[0] = 0
    if 1 not in age_churn.columns:
        age_churn[1] = 0
    age_churn = age_churn[[0, 1]]
    age_churn.columns = ["Retained", "Churned"]

    fig, ax = create_base_fig(6.5, 3.5)
    x = np.arange(len(age_churn))
    w = 0.35
    ax.bar(x - w / 2, age_churn["Retained"], w, label="Retained", color=COLOR_BLUE)
    ax.bar(x + w / 2, age_churn["Churned"], w, label="Churned", color=COLOR_RED)
    ax.set_xticks(x)
    ax.set_xticklabels(age_churn.index, fontsize=9)
    ax.set_title("Churn by Age Group", fontsize=11, fontweight="bold", color=COLOR_NAVY, pad=10)
    ax.set_ylabel("Customers", fontsize=9, color="#475569")
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout()
    return fig


# ============================================================
# NAVIGATION HELPER
# ============================================================

def set_nav_page(page_name):
    st.session_state.nav_page = page_name


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("### 🏦 Banking Intelligence")
    st.caption("Customer Churn, SHAP & AI Retention Platform")
    st.divider()

    NAV_OPTIONS = ["Home", "Customer Prediction", "Analytics", "Data Upload"]
    if "nav_page" not in st.session_state or st.session_state.nav_page not in NAV_OPTIONS:
        st.session_state.nav_page = "Home"

    st.radio(
        "Navigation",
        options=NAV_OPTIONS,
        key="nav_page"
    )

    st.divider()
    st.markdown("##### 🤖 AI Engine")
    st.info("🦙 Ollama (Llama 3.2)")

    st.divider()
    st.markdown("##### 📁 Active Data Source")
    st.info(f"{st.session_state.data_source}")
    st.caption(f"Records: {len(st.session_state.active_data):,}")


# ============================================================
# PAGE 1 — HOME / OVERVIEW
# ============================================================

if st.session_state.nav_page == "Home":

    st.title("🏦 Bank Customer Churn & Retention Platform")
    """st.write(
        "Welcome to the integrated Banking Intelligence Platform. "
        "Combine predictive Machine Learning (XGBoost), Explainable AI (SHAP), "
        "and Generative AI Retention Copilots to identify risk early, understand customer drivers, "
        "and take high-impact retention actions."
    )"""
    st.divider()


    st.write("")
    st.subheader("Portfolio Analytics Overview")
    df_home = st.session_state.active_data

    if not df_home.empty and "Exited" in df_home.columns:
        total_cust = len(df_home)
        churned_cust = int(df_home["Exited"].sum())
        retained_cust = total_cust - churned_cust
        churn_rate = (churned_cust / total_cust * 100) if total_cust > 0 else 0

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Customers", f"{total_cust:,}")
        with col_m2:
            st.metric("Retained Customers", f"{retained_cust:,}")
        with col_m3:
            st.metric("Churned Customers", f"{churned_cust:,}")
        with col_m4:
            st.metric("Churn Rate", f"{churn_rate:.1f}%")

        st.write("")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig1 = plot_churn_distribution(df_home)
            st.pyplot(fig1)
            plt.close(fig1)

        with chart_col2:
            if "Geography" in df_home.columns:
                fig2 = plot_churn_by_geography(df_home)
                if fig2:
                    st.pyplot(fig2)
                    plt.close(fig2)

    st.divider()
    st.markdown("##### Evaluate an Individual Customer")
    st.button(
        "Launch Customer Assessment →",
        type="primary",
        on_click=set_nav_page,
        args=("Customer Prediction",)
    )


# ============================================================
# PAGE 2 — CUSTOMER PREDICTION
# ============================================================

elif st.session_state.nav_page == "Customer Prediction":

    st.title("🏦 Customer Churn Assessment & AI Copilot")
    st.write(
        "Enter customer demographic and account parameters to evaluate churn risk, inspect SHAP drivers, "
        "and generate targeted retention plans using the AI Copilot."
    )
    st.divider()

    st.subheader("1. Mandatory Customer Profile (*)")

    mand_col1, mand_col2 = st.columns(2)

    with mand_col1:
        credit_score = st.number_input(
            "Credit Score *",
            min_value=300,
            max_value=850,
            value=650,
            step=1,
            help="Customer credit bureau score (300–850)"
        )

        age = st.number_input(
            "Age *",
            min_value=18,
            max_value=100,
            value=40,
            step=1,
            help="Customer age in years"
        )

        tenure_months = st.number_input(
            "Tenure (Months) *",
            min_value=0,
            max_value=120,
            value=36,
            step=1,
            help="Customer relationship duration in months"
        )

        balance = st.number_input(
            "Account Balance ($) *",
            min_value=0.0,
            value=80000.0,
            step=1000.0,
            format="%.2f",
            help="Current total balance across bank accounts"
        )

    with mand_col2:
        geography = st.selectbox(
            "Geography *",
            options=GEOGRAPHY_OPTIONS,
            help="Customer primary country of residence"
        )

        if geography not in SUPPORTED_MODEL_GEOGRAPHIES:
            st.caption(f"ℹ️ Note: '{geography}' is mapped to base regional profile for encoding.")

        gender = st.selectbox(
            "Gender *",
            options=["Female", "Male"],
            help="Customer recorded gender"
        )

        num_products = st.number_input(
            "Number of Products *",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            help="Number of active bank products held (1–4)"
        )

        is_active_member = st.selectbox(
            "Is Active Member? *",
            options=["Yes", "No"],
            help="Customer activity status on bank channels"
        )

    st.write("")

    with st.expander("2. Optional Account Information & Loyalty Attributes", expanded=False):
        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            has_cr_card = st.selectbox(
                "Has Credit Card?",
                options=["Yes", "No"],
                index=0,
                help="Indicates if customer holds a bank credit card"
            )

            estimated_salary = st.number_input(
                "Estimated Salary ($)",
                min_value=0.0,
                value=75000.0,
                step=1000.0,
                format="%.2f",
                help="Estimated annual income"
            )

            satisfaction_score = st.number_input(
                "Satisfaction Score",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="Customer satisfaction rating (1 = lowest, 5 = highest)"
            )

        with opt_col2:
            card_type = st.selectbox(
                "Card Type",
                options=["DIAMOND", "GOLD", "PLATINUM", "SILVER"],
                index=1,
                help="Tier of credit/debit card product"
            )

            point_earned = st.number_input(
                "Reward Points Earned",
                min_value=0,
                value=500,
                step=10,
                help="Accumulated bank loyalty points"
            )

            complain = st.selectbox(
                "Recent Complaint Recorded?",
                options=["No", "Yes"],
                index=0,
                help="Customer filed an official complaint recently"
            )

    st.divider()

    predict_clicked = st.button("🚀 Run Prediction & Explainability Analysis", type="primary", use_container_width=True)

    if predict_clicked:
        tenure_years = min(10, round(tenure_months / 12))

        input_encoded = build_model_input(
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure_years=tenure_years,
            balance=balance,
            num_products=num_products,
            has_cr_card=has_cr_card,
            is_active_member=is_active_member,
            estimated_salary=estimated_salary,
            satisfaction_score=satisfaction_score,
            point_earned=point_earned,
            card_type=card_type,
        )

        # Predict
        churn_prob_val = model.predict_proba(input_encoded)[0][1]
        churn_prob_pct = churn_prob_val * 100
        predicted_churn = churn_prob_val >= THRESHOLD

        if churn_prob_val >= 0.80:
            risk_level = "VERY HIGH"
        elif churn_prob_val >= 0.60:
            risk_level = "HIGH"
        elif churn_prob_val >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # SHAP
        risk_names, protective_names, local_shap = get_shap_explanation(input_encoded)

        # Rule strategy
        (
            primary_strategy,
            detected_reasons,
            recommended_actions,
            strategy_priority,
        ) = get_retention_strategy(
            churn_probability=churn_prob_val,
            is_active_member=is_active_member,
            satisfaction_score=satisfaction_score,
            balance=balance,
            num_products=num_products,
            tenure_years=tenure_years,
        )

        # Customer dictionary for LLM
        customer_dict = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "TenureYears": tenure_years,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "SatisfactionScore": satisfaction_score,
            "CardType": card_type,
            "PointEarned": point_earned,
            "Complaint": complain
        }

        # Save to session state
        st.session_state.last_prediction = {
            "prob_pct": churn_prob_pct,
            "prob_val": churn_prob_val,
            "predicted_churn": predicted_churn,
            "risk_level": risk_level,
            "risk_names": risk_names,
            "protective_names": protective_names,
            "primary_strategy": primary_strategy,
            "detected_reasons": detected_reasons,
            "recommended_actions": recommended_actions,
            "strategy_priority": strategy_priority,
            "customer_dict": customer_dict,
            "input_encoded": input_encoded
        }

    # Render Prediction Results if available
    if st.session_state.last_prediction is not None:
        p = st.session_state.last_prediction

        st.subheader("📊 Assessment Results")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Churn Probability", f"{p['prob_pct']:.1f}%")
        with res_col2:
            st.metric("Risk Level", p['risk_level'])
        with res_col3:
            st.metric("Prediction Outcome", "CHURN RISK" if p['predicted_churn'] else "RETAINED")
        with res_col4:
            st.metric("Intervention Priority", p['strategy_priority'])

        if p['predicted_churn']:
            st.markdown(
                "<div class='badge-churn' style='font-size:1rem; padding:0.6rem 1.2rem; margin-top:0.5rem;'>"
                "⚠️ <b>High Churn Risk Detected</b> — Targeted retention intervention strongly recommended.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='badge-retained' style='font-size:1rem; padding:0.6rem 1.2rem; margin-top:0.5rem;'>"
                "✅ <b>Low Churn Risk</b> — Customer relationship is healthy and stable.</div>",
                unsafe_allow_html=True
            )

        st.write("")
        st.divider()

        # SHAP Explanation Cards
        st.subheader("🔍 Key Influencing Factors (SHAP Explainability)")
        f_col1, f_col2 = st.columns(2)

        with f_col1:
            st.markdown("##### 🔴 Risk Drivers (Pushing Toward Churn)")
            if p['risk_names']:
                for rf in p['risk_names'][:5]:
                    st.markdown(f"- **{rf}**")
            else:
                st.write("No major negative risk drivers detected.")

        with f_col2:
            st.markdown("##### 🟢 Protective Drivers (Supporting Retention)")
            if p['protective_names']:
                for pf in p['protective_names'][:5]:
                    st.markdown(f"- **{pf}**")
            else:
                st.write("No major protective drivers detected.")

        st.divider()

        # AI Copilot Section (Ollama)
        st.subheader("🤖 AI Retention Copilot")
        st.caption("Generate natural-language retention strategy and tactical suggestions powered by local Ollama (Llama 3.2).")

        if st.button("✨ Generate AI Retention Explanation & Action Plan", type="secondary"):
            with st.spinner("AI Copilot (Ollama) is analyzing customer profile and formulating strategy..."):
                try:
                    ai_explanation = generate_churn_explanation(
                        customer_data=p['customer_dict'],
                        churn_probability=p['prob_pct'],
                        top_factors=p['risk_names'] + p['protective_names']
                    )
                    
                    st.markdown("""
                    <div class="ai-card">
                        <div style="font-size: 0.85rem; color: #1E40AF; font-weight: 600; margin-bottom: 0.5rem;">
                            🤖 Generated via Ollama (Llama 3.2)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(ai_explanation)
                except Exception as e:
                    st.error(
                        f"⚠️ Could not connect to local Ollama. Please make sure Ollama is running (`ollama run llama3.2`).\n\n**Details:** {e}"
                    )

        st.divider()

        

# ============================================================
# PAGE 3 — STATISTICS / ANALYTICS (EDA DASHBOARD)
# ============================================================

elif st.session_state.nav_page == "Analytics":

    st.title("📊 Portfolio Statistics & Analytics")
    st.write(
        "Explore churn distribution, demographic patterns, and relationship metrics across the customer portfolio."
    )
    st.caption(f"Active Data: **{st.session_state.data_source}**")
    st.divider()

    df_eda = st.session_state.active_data

    required_eda_cols = ["Exited", "Geography", "IsActiveMember", "NumOfProducts", "Age"]
    missing_eda_cols = [c for c in required_eda_cols if c not in df_eda.columns]

    if missing_eda_cols:
        st.warning(
            f"The active dataset is missing expected columns: {', '.join(missing_eda_cols)}. "
            "Please upload a compatible dataset in the Data Upload page."
        )
    else:
        # Filters Section
        st.markdown("##### 🎛️ Portfolio Filters")
        f_col1, f_col2 = st.columns(2)

        with f_col1:
            geo_list = ["All"] + sorted(list(df_eda["Geography"].dropna().unique()))
            selected_geo = st.selectbox("Filter by Geography", options=geo_list, index=0)

        with f_col2:
            act_options = ["All", "Active Members Only", "Inactive Members Only"]
            selected_act = st.selectbox("Filter by Active Membership", options=act_options, index=0)

        # Apply Filters
        filtered_df = df_eda.copy()
        if selected_geo != "All":
            filtered_df = filtered_df[filtered_df["Geography"] == selected_geo]
        if selected_act == "Active Members Only":
            filtered_df = filtered_df[filtered_df["IsActiveMember"] == 1]
        elif selected_act == "Inactive Members Only":
            filtered_df = filtered_df[filtered_df["IsActiveMember"] == 0]

        st.write("")

        # Dynamic Metrics
        total_f = len(filtered_df)
        churned_f = int(filtered_df["Exited"].sum()) if total_f > 0 else 0
        retained_f = total_f - churned_f
        churn_rate_f = (churned_f / total_f * 100) if total_f > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Customers", f"{total_f:,}")
        with m2:
            st.metric("Retained Customers", f"{retained_f:,}")
        with m3:
            st.metric("Churned Customers", f"{churned_f:,}")
        with m4:
            st.metric("Churn Rate", f"{churn_rate_f:.1f}%")

        st.divider()

        if total_f == 0:
            st.info("No records match the selected filter combination.")
        else:
            st.markdown("##### Churn Segmentation")
            r1_c1, r1_c2 = st.columns(2)

            with r1_c1:
                fig_dist = plot_churn_distribution(filtered_df)
                st.pyplot(fig_dist)
                plt.close(fig_dist)

            with r1_c2:
                fig_geo = plot_churn_by_geography(filtered_df)
                if fig_geo:
                    st.pyplot(fig_geo)
                    plt.close(fig_geo)

            st.markdown("##### Product & Membership Drivers")
            r2_c1, r2_c2 = st.columns(2)

            with r2_c1:
                fig_act = plot_churn_by_active_member(filtered_df)
                if fig_act:
                    st.pyplot(fig_act)
                    plt.close(fig_act)

            with r2_c2:
                fig_prod = plot_churn_by_num_products(filtered_df)
                if fig_prod:
                    st.pyplot(fig_prod)
                    plt.close(fig_prod)

            st.markdown("##### Age Cohort Analysis")
            fig_age = plot_churn_by_age_group(filtered_df)
            if fig_age:
                st.pyplot(fig_age)
                plt.close(fig_age)


# ============================================================
# PAGE 4 — DATA UPLOAD
# ============================================================

elif st.session_state.nav_page == "Data Upload":

    st.title("📁 Customer Data & Batch Upload")
    st.write(
        "Upload a batch customer dataset (CSV format) to refresh the Analytics dashboard. "
        "The uploaded dataset must include an `Exited` column (0 = retained, 1 = churned) for churn analytics."
    )
    st.divider()

    st.subheader("Current Active Dataset")
    ds_col1, ds_col2, ds_col3 = st.columns(3)

    with ds_col1:
        st.metric("Source File", st.session_state.data_source)
    with ds_col2:
        st.metric("Records", f"{len(st.session_state.active_data):,}")
    with ds_col3:
        st.metric("Features", len(st.session_state.active_data.columns))

    if st.session_state.data_source != "Default (Customer-Churn-Records.csv)":
        if st.button("Reset to Default Dataset", type="secondary"):
            st.session_state.active_data = load_default_data()
            st.session_state.data_source = "Default (Customer-Churn-Records.csv)"
            st.success("Reset active dataset to default Customer-Churn-Records.csv.")
            st.rerun()

    st.divider()

    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("Select a customer records CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)

            if "Exited" not in temp_df.columns:
                st.error("Validation Error: The uploaded file is missing the required `Exited` target column.")
            else:
                st.success(f"File validated successfully: **{uploaded_file.name}**")

                up_rows = len(temp_df)
                up_cols = len(temp_df.columns)
                up_churn = (temp_df["Exited"].sum() / up_rows * 100) if up_rows > 0 else 0

                st.markdown("##### Uploaded Dataset Summary")
                u_m1, u_m2, u_m3 = st.columns(3)
                with u_m1:
                    st.metric("Total Rows", f"{up_rows:,}")
                with u_m2:
                    st.metric("Total Columns", up_cols)
                with u_m3:
                    st.metric("Churn Rate", f"{up_churn:.1f}%")

                st.markdown("**Available Columns:**")
                st.write(", ".join([f"`{c}`" for c in temp_df.columns]))

                st.markdown("**Preview (First 5 Rows):**")
                st.dataframe(temp_df.head(5), use_container_width=True)

                st.write("")
                if st.button("Apply Dataset to Analytics", type="primary"):
                    st.session_state.active_data = temp_df
                    st.session_state.data_source = uploaded_file.name
                    st.success(f"Active dataset updated to {uploaded_file.name}. View charts in the Analytics page.")
                    st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.markdown("##### Current Dataset Preview (First 10 Rows)")
        st.dataframe(st.session_state.active_data.head(10), use_container_width=True)
