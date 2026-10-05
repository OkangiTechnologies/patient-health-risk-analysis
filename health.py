"""
STEP 7 — STREAMLIT DASHBOARD
==============================
Patient Health Records — Has_Disease prediction

Run with:  streamlit run app.py

Tabs:
  Overview          - what the project is about, headline finding stated up front
  Data Explorer      - filters/charts over the cleaned patient data
  Model Performance  - full model comparison, confusion matrix, ROC curves, error analysis
  Live Prediction    - user enters values -> the saved pipeline returns a prediction

The model is loaded once from disk (joblib) and NEVER retrained here.

IMPORTANT — carried over from the analysis notebooks (Sections 14-15):
this model performs at chance level (ROC-AUC ~ 0.50) on this dataset. That
finding is surfaced prominently in every tab, not just footnoted, because
hiding it would misrepresent what the app can actually tell a user.
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")  # sklearn inter-version pickle warnings only; predict/predict_proba are unaffected

st.set_page_config(page_title="Patient Health Records — Disease Prediction", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Load everything once, cached
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    pipeline = joblib.load("models/final_model_pipeline.joblib")
    feature_cols = joblib.load("models/model_feature_columns.joblib")
    comparison = pd.read_csv("models/model_comparison.csv")
    with open("models/test_report.json") as f:
        test_report = json.load(f)
    return pipeline, feature_cols, comparison, test_report


@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_patient_data.csv")
    scored = pd.read_csv("data/unlabeled_patients_scored.csv")
    return df, scored


pipeline, feature_cols, comparison, test_report = load_model_artifacts()
df, scored = load_data()

st.title("🩺 Patient Health Records — Disease Prediction")
st.caption("10,000 patient records · predicting Has_Disease from demographic, vital-sign, and medication features.")

st.error(
    "**Headline finding, stated up front:** every model tested — 5 model families, 2 preprocessing "
    "pipelines, with cross-validation and hyperparameter tuning — lands at **ROC-AUC ≈ 0.50**, "
    "i.e. indistinguishable from a coin flip. The features available in this dataset carry no "
    "measurable signal for `Has_Disease`. This dashboard is a complete, working deployment demo "
    "(load → explore → predict), not a tool that should inform real clinical decisions. See the "
    "**Model Performance** tab for the full evidence.",
    icon="⚠️",
)

tab_overview, tab_explore, tab_perf, tab_predict = st.tabs(
    ["📋 Overview", "🔎 Data Explorer", "📈 Model Performance", "🔮 Live Prediction"]
)

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
with tab_overview:
    st.header("What this project is about")
    st.markdown(
        """
**The question:** can a patient's demographic details, vital signs, and current
medications predict whether they have a disease (`Has_Disease`)?

**The data:** 10,000 synthetic-but-messy patient records — every column except
Name had missing values, `Age` and `Heart_Rate` hid bad data as invalid
sentinel values (e.g. Age = 250, Heart Rate = 0) instead of blanks, and
categorical fields needed unifying from inconsistent raw labels. About half
the rows (5,026) have no `Has_Disease` label at all — those were reserved as
a realistic "new patient" deployment scenario rather than used for training
or evaluation.

**Type of problem:** binary **classification** — classes are close to
balanced (2,513 No / 2,461 Yes among labeled rows), so the imbalance
problems that show up in some datasets aren't the issue here.

**What was tried:** Logistic Regression, KNN, SVM, Random Forest, and
Gradient Boosting, across 2 preprocessing pipelines, with 5-fold
cross-validation and grid-search tuning on the top performers.

**What was found:** nothing worked, and that's the actual result. Every
model's test ROC-AUC sits at ~0.48–0.51 — statistically indistinguishable
from guessing. Error analysis (by gender, city, smoking status, age band)
found no subgroup where the model does meaningfully better, and feature
importances are flat rather than dominated by any real predictor. This
isn't a case of picking the wrong algorithm; the available columns simply
don't contain the information needed to predict this target.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(df):,}")
    c2.metric("Labeled rows", f"{df['Has_Disease'].notna().sum():,}")
    c3.metric("Final model", test_report["final_model_name"])
    c4.metric("Test ROC-AUC", f"{test_report['roc_auc']:.3f}", help="0.50 = chance level")

# ===========================================================================
# TAB 2 — DATA EXPLORER
# ===========================================================================
with tab_explore:
    st.header("Explore the patient data")

    with st.sidebar:
        st.subheader("Filters")
        gender_filter = st.multiselect(
            "Gender", sorted(df["Gender"].dropna().unique()),
            default=sorted(df["Gender"].dropna().unique())
        )
        city_filter = st.multiselect(
            "City", sorted(df["City"].dropna().unique()),
            default=sorted(df["City"].dropna().unique())
        )
        label_filter = st.selectbox(
            "Has_Disease status", ["All rows", "Labeled only", "Unlabeled only"]
        )

    fdf = df[df["Gender"].isin(gender_filter) & df["City"].isin(city_filter)]
    if label_filter == "Labeled only":
        fdf = fdf[fdf["Has_Disease"].notna()]
    elif label_filter == "Unlabeled only":
        fdf = fdf[fdf["Has_Disease"].isna()]

    st.caption(f"Showing {len(fdf):,} of {len(df):,} patients matching filters")

    col1, col2 = st.columns(2)
    with col1:
        num_feature = st.selectbox(
            "Numeric feature to plot",
            ["Age", "BMI", "Heart_Rate", "BP_Systolic", "BP_Diastolic",
             "Follow_Up", "Days_Since_Last_Visit", "Medication_Count"],
        )
        plot_df = fdf.copy()
        plot_df["Has_Disease_label"] = plot_df["Has_Disease"].map({0: "No", 1: "Yes"}).fillna("Unlabeled")
        fig = px.histogram(
            plot_df, x=num_feature, color="Has_Disease_label", barmode="overlay",
            opacity=0.6, nbins=30,
            color_discrete_map={"No": "#4C956C", "Yes": "#C0392B", "Unlabeled": "#B0B0B0"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Notice how similar the 'No' and 'Yes' distributions look for {num_feature} — "
            "this visual overlap is exactly what a ROC-AUC near 0.50 looks like."
        )

    with col2:
        cat_feature = st.selectbox(
            "Categorical feature to plot",
            ["Gender", "City", "Cholesterol_Level", "Diabetic", "Smoker"],
        )
        ct = fdf.groupby(cat_feature)["Has_Disease"].agg(["mean", "count"]).reset_index()
        ct.columns = [cat_feature, "Disease rate", "N"]
        fig2 = px.bar(
            ct, x=cat_feature, y="Disease rate", text="N",
            color_discrete_sequence=["#4C6EF5"],
        )
        fig2.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig2.add_hline(y=fdf["Has_Disease"].mean(), line_dash="dash", line_color="gray",
                        annotation_text="overall rate")
        fig2.update_layout(height=420, yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Disease rate by {cat_feature} — bars sitting close to the dashed "
                   "overall-rate line mean this feature isn't separating the classes.")

    st.subheader("Raw filtered rows")
    st.dataframe(fdf.drop(columns=["Name"]).head(300), height=350)

# ===========================================================================
# TAB 3 — MODEL PERFORMANCE
# ===========================================================================
with tab_perf:
    st.header("How the models perform")

    st.subheader("All 5 models — test set")
    st.dataframe(
        comparison.style.format({
            "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "roc_auc": "{:.3f}",
        }).background_gradient(subset=["roc_auc"], cmap="RdYlGn", vmin=0.45, vmax=0.55),
        use_container_width=True,
    )
    st.caption(
        "ROC-AUC is the primary metric (classes are close to balanced here, so accuracy is "
        "meaningful too) — 0.50 is chance level. Every model, tuned or not, sits within a few "
        "points of 0.50: this is a flat, uniform result across model families, not one model "
        "underperforming the rest."
    )

    st.subheader(f"Final deployed model: {test_report['final_model_name']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{test_report['accuracy']:.3f}")
    c2.metric("Precision", f"{test_report['precision']:.3f}")
    c3.metric("Recall", f"{test_report['recall']:.3f}")
    c4.metric("ROC-AUC", f"{test_report['roc_auc']:.3f}", help="0.50 = chance level")
    st.caption(test_report["note"])

    st.image("figures/fig_eval.png",
              caption="Confusion matrix (tuned Random Forest) and ROC curves for all 5 models — test set")

    st.subheader("Error analysis")
    st.image("figures/fig_error_analysis.png",
              caption="Predicted-probability distribution by true class — near-total overlap is expected at chance level")
    st.markdown(
        "The notebook's subgroup breakdown (gender, city, smoking status, age band) found no "
        "subgroup with a meaningfully different accuracy from the overall ~49% rate, and feature "
        "importances were flat rather than dominated by any real predictor — consistent evidence "
        "that this isn't a fixable modeling problem, it's a data problem."
    )

    st.subheader("Missing data")
    st.image("figures/fig_missing.png", caption="Missing-value rate by column, before cleaning")

    st.warning(
        "**Is it good enough?** No. An ROC-AUC of ~0.50 means the model provides zero "
        "discriminative value beyond guessing — it should not be used to inform any real "
        "clinical decision. What it does demonstrate is a complete, correctly-built deployment "
        "pipeline (cleaning → features → CV/tuning → held-out evaluation → persisted model → "
        "live scoring) that would be genuinely useful the moment the input data contains real "
        "predictive signal (e.g. lab trends, family history, leakage-free comorbidity codes)."
    )

# ===========================================================================
# TAB 4 — LIVE PREDICTION
# ===========================================================================
with tab_predict:
    st.header("Get a live prediction")
    st.caption("Fill in patient details — the saved pipeline predicts instantly, no retraining.")
    st.info(
        "Reminder: this model performs at chance level. Treat any prediction below as a "
        "demonstration of the deployment mechanics, not a real health assessment.",
        icon="ℹ️",
    )

    med_cols = [c for c in feature_cols if c.startswith("Med_")]

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=50)
        bmi = st.number_input("BMI", min_value=15.0, max_value=35.0, value=23.0, step=0.1)
        heart_rate = st.number_input("Heart rate (bpm)", min_value=50, max_value=120, value=80)
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
    with col2:
        bp_systolic = st.number_input("Blood pressure - systolic", min_value=90, max_value=160, value=120)
        bp_diastolic = st.number_input("Blood pressure - diastolic", min_value=60, max_value=100, value=80)
        cholesterol = st.selectbox("Cholesterol level", ["Normal", "High", "Unknown"])
        diabetic = st.selectbox("Diabetic", ["No", "Yes", "Unknown"])
    with col3:
        city = st.selectbox("City", ["Boston", "Chicago", "Los Angeles", "New York", "Unknown"])
        smoker = st.selectbox("Smoker status", ["No", "Current", "Former", "Unknown"])
        follow_up = st.number_input("Follow-up interval (days)", min_value=0, max_value=90, value=30)
        days_since_visit = st.number_input("Days since last visit", min_value=0, max_value=3000, value=365)

    st.markdown("**Current medications**")
    med_cols_ui = st.columns(len(med_cols)) if med_cols else []
    med_flags = {}
    for i, mc in enumerate(med_cols):
        drug_name = mc.replace("Med_", "")
        with med_cols_ui[i]:
            med_flags[mc] = int(st.checkbox(drug_name, value=False))
    medication_count = sum(med_flags.values())

    if st.button("Predict", type="primary"):
        row = {c: 0 for c in med_cols}
        row.update(med_flags)
        row.update({
            "Age": age, "Gender": gender, "City": city, "BMI": bmi,
            "Heart_Rate": heart_rate, "Cholesterol_Level": cholesterol,
            "Diabetic": diabetic, "Smoker": smoker, "Follow_Up": follow_up,
            "BP_Systolic": bp_systolic, "BP_Diastolic": bp_diastolic,
            "Days_Since_Last_Visit": days_since_visit,
            "Medication_Count": medication_count,
        })
        X_new = pd.DataFrame([row])[feature_cols]

        pred = pipeline.predict(X_new)[0]
        proba = pipeline.predict_proba(X_new)[0, 1]

        label = "Disease predicted" if pred == 1 else "No disease predicted"
        color = "#C0392B" if pred == 1 else "#4C956C"
        st.markdown(f"### {label}: <span style='color:{color}'>probability {proba:.1%}</span>",
                    unsafe_allow_html=True)

        fig3 = px.bar(
            x=["No Disease", "Disease"], y=[1 - proba, proba],
            color=["No Disease", "Disease"],
            color_discrete_map={"No Disease": "#4C956C", "Disease": "#C0392B"},
            range_y=[0, 1], labels={"x": "", "y": "Probability"},
        )
        fig3.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "This probability sits close to 50% for almost any input you try — that's the "
            "chance-level ROC-AUC in action, not a bug in the form. Changing individual "
            "inputs won't move the prediction the way it would for a model with real signal."
        )
