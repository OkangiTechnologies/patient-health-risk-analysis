"""
predict.py — interactive deployment script for the Patient Health Records model.

Run it with no arguments:
    python predict.py

It will ask you for each value one at a time in the terminal, then print the
prediction. Press Enter on any optional question to accept the default shown
in brackets.

IMPORTANT: as documented in the analysis notebook (Section 14/16), this model's
available features carry no measurable predictive signal for Has_Disease in the
source dataset (ROC-AUC ~= 0.50, i.e. chance level). This script demonstrates a
complete, working deployment path (load -> collect input -> predict), but its
output should NOT be used for real clinical decisions.
"""
import sys
import joblib
import pandas as pd


def load_pipeline(model_path='final_model_pipeline.joblib',
                   columns_path='model_feature_columns.joblib'):
    try:
        pipeline = joblib.load(model_path)
        feature_cols = joblib.load(columns_path)
    except FileNotFoundError as e:
        sys.exit(f"Could not find model artifacts ({e}). Make sure "
                  f"final_model_pipeline.joblib and model_feature_columns.joblib "
                  f"are in the same folder as this script.")
    return pipeline, feature_cols


def ask_number(prompt, required=True, default=None):
    while True:
        raw = input(f"{prompt}{' [' + str(default) + ']' if default is not None else ''}: ").strip()
        if not raw:
            if default is not None:
                return default
            if not required:
                return default
            print("  This value is required, please enter a number.")
            continue
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a valid number.")


def ask_choice(prompt, choices, default):
    choice_str = "/".join(choices)
    while True:
        raw = input(f"{prompt} ({choice_str}) [{default}]: ").strip()
        if not raw:
            return default
        for c in choices:
            if raw.lower() == c.lower():
                return c
        print(f"  Please enter one of: {choice_str}")


def ask_text(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def ask_yes_no(prompt, default=False):
    raw = input(f"{prompt} (y/n) [{'y' if default else 'n'}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith('y')


def collect_patient_data(feature_cols):
    print("=== Enter patient details (press Enter to accept the default in brackets) ===\n")

    age = ask_number("Age", required=True)
    bmi = ask_number("BMI", required=True)
    heart_rate = ask_number("Heart rate (bpm)", required=True)
    bp_systolic = ask_number("Blood pressure - systolic (top number)", required=True)
    bp_diastolic = ask_number("Blood pressure - diastolic (bottom number)", required=True)

    gender = ask_choice("Gender", ["Male", "Female", "Unknown"], "Unknown")
    city = ask_text("City", "Unknown")
    cholesterol = ask_choice("Cholesterol level", ["Normal", "Borderline", "High", "Unknown"], "Unknown")
    diabetic = ask_choice("Diabetic", ["Yes", "No", "Unknown"], "Unknown")
    smoker = ask_choice("Smoker status", ["Current", "Former", "No", "Unknown"], "Unknown")
    follow_up = ask_number("Follow-up interval in days", required=False, default=30)
    days_since_visit = ask_number("Days since last visit", required=False, default=0)

    print("\nCurrent medications (answer y/n for each):")
    med_flags = {}
    for med_col in [c for c in feature_cols if c.startswith('Med_')]:
        drug_name = med_col.replace('Med_', '')
        med_flags[med_col] = int(ask_yes_no(f"  Taking {drug_name}?", default=False))
    medication_count = sum(med_flags.values())

    row = {c: 0 for c in feature_cols if c.startswith('Med_')}
    row.update(med_flags)
    row.update({
        'Age': age,
        'Gender': gender,
        'City': city,
        'BMI': bmi,
        'Heart_Rate': heart_rate,
        'Cholesterol_Level': cholesterol,
        'Diabetic': diabetic,
        'Smoker': smoker,
        'Follow_Up': follow_up,
        'BP_Systolic': bp_systolic,
        'BP_Diastolic': bp_diastolic,
        'Days_Since_Last_Visit': days_since_visit,
        'Medication_Count': medication_count,
    })
    return pd.DataFrame([row])[feature_cols]


def main():
    pipeline, feature_cols = load_pipeline()
    X_new = collect_patient_data(feature_cols)

    pred = pipeline.predict(X_new)[0]
    proba = pipeline.predict_proba(X_new)[0, 1]

    print("\n=== Result ===")
    print(f"Predicted Has_Disease: {int(pred)}")
    print(f"Predicted probability: {proba:.3f}")
    print("\nNote: this model performs at chance level on the source dataset (see notebook "
          "Sections 14-15) -- treat this output as a deployment-mechanics demo, not a clinical signal.")

    again = ask_yes_no("\nScore another patient?", default=False)
    if again:
        print()
        main()


if __name__ == '__main__':
    main()
