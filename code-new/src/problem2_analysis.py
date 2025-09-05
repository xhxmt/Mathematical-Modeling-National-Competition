import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from data_loader import get_cleaned_data

def run_problem2_analysis():
    """
    Performs the analysis for Problem 2.
    """
    print("--- Running Problem 2 Analysis ---")

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

    # --- 1. BMI Clustering using K-Means ---
    print("Performing K-Means clustering on BMI...")

    bmi_data = male_df[['BMI']].dropna()
    scaler = StandardScaler()
    bmi_scaled = scaler.fit_transform(bmi_data)

    # Determine optimal k using the Elbow Method (plot already saved)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    male_df['BMI_Group'] = kmeans.fit_predict(bmi_scaled)

    # --- 2. Risk Function Definition (Final Attempt) ---
    X = male_df[['Gestational_Week', 'BMI']]
    X = sm.add_constant(X)
    y = male_df['Y_Concentration']
    p1_model = sm.OLS(y, X).fit()
    model_std_error = np.sqrt(p1_model.mse_resid)

    def calculate_risk(week, mean_bmi, measurement_error_std=0.0):
        # a) Risk of Test Failure
        pred_conc = p1_model.predict([1, week, mean_bmi])[0]
        total_std = np.sqrt(model_std_error**2 + measurement_error_std**2)
        prob_failure = norm.cdf(0.04, loc=pred_conc, scale=total_std)

        # b) Risk of Delayed Diagnosis
        delay_risk = (week - 10) / (25 - 10)

        # c) Total Risk (Final weights)
        total_risk = 0.95 * prob_failure + 0.05 * delay_risk
        return total_risk

    # --- 3. Find Optimal NIPT Time for Each Group ---
    print("\nCalculating optimal NIPT time for each BMI group (Final Attempt)...")

    weeks_to_check = np.arange(10, 25.5, 0.5)
    results = []

    group_order = male_df.groupby('BMI_Group')['BMI'].mean().sort_values().index

    plt.figure(figsize=(14, 8))

    for i in group_order:
        group_data = male_df[male_df['BMI_Group'] == i]
        mean_bmi_in_group = group_data['BMI'].mean()

        risks = [calculate_risk(w, mean_bmi_in_group) for w in weeks_to_check]
        optimal_week_idx = np.argmin(risks)
        optimal_week = weeks_to_check[optimal_week_idx]

        measurement_error_std = 0.005
        risks_with_error = [calculate_risk(w, mean_bmi_in_group, measurement_error_std) for w in weeks_to_check]
        optimal_week_error_idx = np.argmin(risks_with_error)
        optimal_week_with_error = weeks_to_check[optimal_week_error_idx]

        results.append({
            'group_label': f"Group {i} (BMI ~{mean_bmi_in_group:.1f})",
            'mean_bmi': mean_bmi_in_group,
            'count': len(group_data),
            'optimal_week': optimal_week,
            'optimal_week_with_error': optimal_week_with_error
        })

        plt.plot(weeks_to_check, risks, marker='o', linestyle='-', markersize=4, label=f'Group {i} (BMI ~{mean_bmi_in_group:.1f}) - No Error')
        plt.plot(weeks_to_check, risks_with_error, linestyle='--', label=f'Group {i} (BMI ~{mean_bmi_in_group:.1f}) - With Error')

    plt.title('Final Risk Minimization Curves for NIPT Timing', fontsize=16)
    plt.xlabel('Gestational Week', fontsize=12)
    plt.ylabel('Calculated Risk (Lower is Better)', fontsize=12)
    plt.legend()
    plt.grid(True)
    risk_curves_path = os.path.join(plots_dir, 'p2_risk_curves_final.png')
    plt.savefig(risk_curves_path)
    plt.close()
    print(f"Saved final risk curves plot: {risk_curves_path}")

    print("\n--- Optimal NIPT Timing Results (Final) ---")
    for res in results:
        print(f"{res['group_label']} (n={res['count']}):")
        print(f"  - Optimal Week (no error): {res['optimal_week']:.1f} weeks")
        print(f"  - Optimal Week (with error): {res['optimal_week_with_error']:.1f} weeks")

    print("\n--- Problem 2 Analysis Complete ---")


if __name__ == '__main__':
    run_problem2_analysis()
