from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "jobs.csv"
MODEL_PATH = BASE_DIR / "models" / "salary_model.pkl"

st.set_page_config(page_title="JobMarketAI Dashboard", layout="wide")

st.title("JobMarketAI")
st.subheader("Job Market Intelligence and Salary Prediction Platform")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["salary_avg"] = (df["salary_min"] + df["salary_max"]) / 2
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

st.markdown("---")

st.header("Dataset Overview")
st.write("Preview of the job market dataset:")
st.dataframe(df)

col1, col2, col3 = st.columns(3)
col1.metric("Total Jobs", len(df))
col2.metric("Locations", df["location"].nunique())
col3.metric("Average Salary", f"{df['salary_avg'].mean():,.0f}")

st.markdown("---")

st.header("Top Skills in Demand")

df["skills_list"] = df["skills"].str.split(",")
all_skills = df["skills_list"].explode().str.strip()
skill_counts = all_skills.value_counts()

fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(x=skill_counts.values, y=skill_counts.index, ax=ax1)
ax1.set_title("Most Demanded Skills")
ax1.set_xlabel("Number of Job Postings")
ax1.set_ylabel("Skill")
st.pyplot(fig1)

st.markdown("---")

st.header("Salary Insights")

tab1, tab2, tab3 = st.tabs(["Salary Distribution", "By Role", "By Location"])

with tab1:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["salary_avg"], bins=10, ax=ax2)
    ax2.set_title("Salary Distribution")
    ax2.set_xlabel("Average Salary")
    ax2.set_ylabel("Number of Jobs")
    st.pyplot(fig2)

with tab2:
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="job_title", y="salary_avg", ax=ax3)
    ax3.set_title("Salary by Job Role")
    ax3.tick_params(axis="x", rotation=45)
    st.pyplot(fig3)

with tab3:
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="location", y="salary_avg", ax=ax4)
    ax4.set_title("Average Salary by Location")
    ax4.tick_params(axis="x", rotation=45)
    st.pyplot(fig4)

st.markdown("---")

st.header("Salary Prediction")

job_title = st.selectbox("Select Job Title", sorted(df["job_title"].unique()))
location = st.selectbox("Select Location", sorted(df["location"].unique()))
experience_level = st.selectbox("Select Experience Level", sorted(df["experience_level"].unique()))

if st.button("Predict Salary"):
    training_features = df[["job_title", "location", "experience_level"]]
    training_features = pd.get_dummies(training_features)

    sample = pd.DataFrame({
        "job_title": [job_title],
        "location": [location],
        "experience_level": [experience_level]
    })

    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=training_features.columns, fill_value=0)

    prediction = model.predict(sample)[0]
    st.success(f"Predicted Average Salary: {prediction:,.2f}")