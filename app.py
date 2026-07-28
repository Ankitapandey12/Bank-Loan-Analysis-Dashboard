import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Page settings
st.set_page_config(
    page_title="Bank Loan Analysis Dashboard",
    layout="wide"
)


# ================= BACKGROUND =================

st.markdown(
    """
    <style>

    .stApp {
        background-image: url("https://images.unsplash.com/photo-1554224155-6726b3ff858f");
        background-size: cover;
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(255,255,255,0.90);
        padding: 2rem;
        border-radius: 15px;
    }

    h1 {
        text-align:center;
        color:#1f4e79;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================= LOAD DATA =================

@st.cache_data
def load_data():
    df = pd.read_csv("data/bank_loan.csv")
    return df


df = load_data()


# ================= TITLE =================

st.title("🏦 Bank Loan Analysis Dashboard")

st.write(
    "Interactive analysis of bank loan data"
)


# ================= KPI SECTION =================

st.header("📊 Key Performance Indicators")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Applications",
        f"{len(df):,}"
    )


with col2:
    st.metric(
        "Average Income",
        f"£{df['person_income'].mean():,.0f}"
    )


with col3:
    st.metric(
        "Average Loan Amount",
        f"£{df['loan_amnt'].mean():,.0f}"
    )


with col4:

    if "loan_status" in df.columns:

        approval_rate = (
            df["loan_status"]
            .value_counts(normalize=True)
            .get(1,0)*100
        )

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )



# ================= DATASET OVERVIEW =================

st.header("📋 Dataset Overview")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Total Rows",
        df.shape[0]
    )


with col2:
    st.metric(
        "Total Features",
        df.shape[1]
    )


with col3:
    st.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )
    # ================= SIDEBAR FILTER =================

st.sidebar.header("🔍 Filters")


filtered_df = df.copy()


if "loan_status" in df.columns:

    status = st.sidebar.selectbox(
        "Select Loan Status",
        df["loan_status"].unique()
    )


    filtered_df = df[
        df["loan_status"] == status
    ]



# ================= DATA PREVIEW =================

st.header("📄 Dataset Preview")


st.dataframe(
    filtered_df.head(10),
    use_container_width=True
)



# ================= LOAN STATUS ANALYSIS =================

if "loan_status" in df.columns:

    st.header("📈 Loan Status Analysis")


    fig, ax = plt.subplots(figsize=(7,4))


    sns.countplot(
        data=df,
        x="loan_status",
        ax=ax
    )


    ax.set_title(
        "Loan Approval Status"
    )


    st.pyplot(fig)



# ================= LOAN AMOUNT ANALYSIS =================

st.header("💰 Loan Amount Distribution")


fig, ax = plt.subplots(figsize=(8,4))


sns.histplot(
    df["loan_amnt"],
    bins=30,
    kde=True,
    ax=ax
)


ax.set_title(
    "Distribution of Loan Amount"
)


st.pyplot(fig)



# ================= INCOME VS LOAN =================

st.header("💵 Income vs Loan Amount")


fig, ax = plt.subplots(figsize=(8,5))


sns.scatterplot(
    data=df,
    x="person_income",
    y="loan_amnt",
    hue="loan_status",
    ax=ax
)


st.pyplot(fig)
# ================= CUSTOMER ANALYSIS =================

st.header("👥 Customer Analysis")


# Gender Analysis

if "person_gender" in df.columns:

    st.subheader("Gender Wise Loan Distribution")


    fig, ax = plt.subplots(figsize=(6,4))


    sns.countplot(
        data=df,
        x="person_gender",
        ax=ax
    )


    ax.set_title(
        "Loan Applicants by Gender"
    )


    st.pyplot(fig)



# Education Analysis

if "person_education" in df.columns:

    st.subheader("Education Wise Loan Distribution")


    fig, ax = plt.subplots(figsize=(8,4))


    sns.countplot(
        data=df,
        x="person_education",
        ax=ax
    )


    ax.set_title(
        "Loan Applicants by Education"
    )


    plt.xticks(rotation=45)


    st.pyplot(fig)



# Loan Intent Analysis

if "loan_intent" in df.columns:

    st.subheader("Loan Purpose Analysis")


    fig, ax = plt.subplots(figsize=(8,4))


    sns.countplot(
        data=df,
        y="loan_intent",
        ax=ax
    )


    ax.set_title(
        "Purpose of Taking Loan"
    )


    st.pyplot(fig)



# Home Ownership Analysis

if "person_home_ownership" in df.columns:

    st.subheader("Home Ownership Analysis")


    fig, ax = plt.subplots(figsize=(7,4))


    sns.countplot(
        data=df,
        x="person_home_ownership",
        ax=ax
    )


    ax.set_title(
        "Applicants by Home Ownership"
    )


    st.pyplot(fig)




# ================= APPROVAL ANALYSIS =================

st.header("✅ Loan Approval Analysis")


if "loan_status" in df.columns:


    # Gender Approval

    if "person_gender" in df.columns:

        st.subheader("Loan Approval by Gender")


        fig, ax = plt.subplots(figsize=(7,4))


        sns.countplot(
            data=df,
            x="person_gender",
            hue="loan_status",
            ax=ax
        )


        ax.set_title(
            "Gender Wise Loan Approval"
        )


        st.pyplot(fig)



    # Education Approval

    if "person_education" in df.columns:

        st.subheader("Education vs Loan Status")


        fig, ax = plt.subplots(figsize=(8,4))


        sns.countplot(
            data=df,
            x="person_education",
            hue="loan_status",
            ax=ax
        )


        plt.xticks(rotation=45)


        ax.set_title(
            "Education Wise Approval"
        )


        st.pyplot(fig)




# ================= CORRELATION =================

st.header("🔥 Feature Correlation")


numeric_df = df.select_dtypes(
    include=["int64","float64"]
)


fig, ax = plt.subplots(figsize=(10,6))


sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)


ax.set_title(
    "Correlation Heatmap"
)


st.pyplot(fig)




# ================= DASHBOARD SUMMARY =================

st.header("📋 Dashboard Summary")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Dataset Information")


    st.write(
        f"""
        - Total Records: {df.shape[0]}
        - Total Features: {df.shape[1]}
        - Missing Values: {df.isnull().sum().sum()}
        """
    )



with col2:

    st.subheader("Loan Information")


    st.write(
        f"""
        - Average Income: £{df['person_income'].mean():,.0f}
        - Average Loan Amount: £{df['loan_amnt'].mean():,.0f}
        - Maximum Loan Amount: £{df['loan_amnt'].max():,.0f}
        """
    )



# ================= STATISTICAL SUMMARY =================

st.header("📈 Statistical Summary")


st.write(
    df.describe()
)