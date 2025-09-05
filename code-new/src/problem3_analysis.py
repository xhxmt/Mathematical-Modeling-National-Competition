import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from data_loader import get_cleaned_data

def run_problem3_analysis():
    """
    Performs the analysis for Problem 3 using an XGBoost model.
    """
    print("--- Running Problem 3 Analysis ---")

    # --- 0. Load Data & Setup ---
    try:
        male_df, _ = get_cleaned_data()
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return

    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # --- 1. Train a More Powerful Model (XGBoost) ---
    print("Training XGBoost model...")

    features = ['Gestational_Week', 'BMI', 'Age', 'Height', 'Weight']
    target = 'Y_Concentration'

    X = male_df[features]
    y = male_df[target]

    # Split data for training and to calculate residuals
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost Regressor
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgbr.fit(X_train, y_train)

    # --- 2. Evaluate Model and Estimate Prediction Error ---
    # Use residuals from the test set to estimate prediction error standard deviation
    y_pred = xgbr.predict(X_test)
    residuals = y_test - y_pred
    prediction_std_error = np.std(residuals)
    print(f"Estimated prediction standard error from XGBoost residuals: {prediction_std_error:.4f}")

    # --- 3. Feature Importance ---
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgbr, height=0.8)
    plt.title('XGBoost Feature Importance', fontsize=16)
    plt.xlabel('F-score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    feature_imp_path = os.path.join(plots_dir, 'p3_feature_importance.png')
    plt.savefig(feature_imp_path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot: {feature_imp_path}")

    # --- 4. Determine Optimal Time based on Success Probability ---
    print("\nCalculating optimal NIPT time for each BMI group (XGBoost model)...")

    # Use the same BMI groups as in Problem 2
    bmi_data = male_df[['BMI']].dropna()
    scaler = StandardScaler()
    bmi_scaled = scaler.fit_transform(bmi_data)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    male_df['BMI_Group'] = kmeans.fit_predict(bmi_scaled)
    group_order = male_df.groupby('BMI_Group')['BMI'].mean().sort_values().index

    weeks_to_check = np.arange(10, 25.5, 0.5)
    results = []

    plt.figure(figsize=(14, 8))

    for i in group_order:
        group_data = male_df[male_df['BMI_Group'] == i]
        # Create a representative patient profile for the group
        profile = {
            'BMI': group_data['BMI'].mean(),
            'Age': group_data['Age'].mean(),
            'Height': group_data['Height'].mean(),
            'Weight': group_data['Weight'].mean()
        }

        success_probs = []
        for week in weeks_to_check:
            profile['Gestational_Week'] = week
            profile_df = pd.DataFrame([profile])

            # Predict concentration
            pred_conc = xgbr.predict(profile_df[features])[0]

            # Calculate probability of success (Y_Concentration >= 4%)
            prob_success = 1 - norm.cdf(0.04, loc=pred_conc, scale=prediction_std_error)
            success_probs.append(prob_success)

        # Find the first week where success probability is >= 95%
        try:
            optimal_week_idx = np.where(np.array(success_probs) >= 0.95)[0][0]
            optimal_week = weeks_to_check[optimal_week_idx]
        except IndexError:
            # If 95% is never reached, recommend the week with max probability
            optimal_week = weeks_to_check[np.argmax(success_probs)]

        results.append({
            'group_label': f"Group {i} (BMI ~{profile['BMI']:.1f})",
            'optimal_week': optimal_week
        })

        plt.plot(weeks_to_check, success_probs, marker='o', linestyle='-', markersize=4, label=f"Group {i} (BMI ~{profile['BMI']:.1f})")

    plt.title('Success Probability Curves (Y-Conc >= 4%) by BMI Group', fontsize=16)
    plt.xlabel('Gestational Week', fontsize=12)
    plt.ylabel('Probability of Successful Test', fontsize=12)
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Success Threshold')
    plt.legend()
    plt.grid(True)
    success_curves_path = os.path.join(plots_dir, 'p3_success_curves.png')
    plt.savefig(success_curves_path)
    plt.close()
    print(f"Saved success probability curves plot: {success_curves_path}")

    print("\n--- Optimal NIPT Timing Results (XGBoost) ---")
    for res in results:
        print(f"{res['group_label']}: Recommended Week = {res['optimal_week']:.1f}")

    print("\n--- Problem 3 Analysis Complete ---")

if __name__ == '__main__':
    run_problem3_analysis()
