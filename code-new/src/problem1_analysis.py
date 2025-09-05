import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import get_cleaned_data

def run_problem1_analysis():
    """
    Performs the analysis for Problem 1:
    1. Loads the data.
    2. Creates and saves scatter plots for EDA.
    3. Builds and prints the summary of a multiple linear regression model.
    """
    print("--- Running Problem 1 Analysis ---")

    # Load data
    try:
        male_df, _ = get_cleaned_data()
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return

    # Define the output directory for plots
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # --- 1. Exploratory Data Analysis (EDA) ---
    print("Generating and saving plots...")

    # Plot 1: Y-Concentration vs. Gestational Week
    plt.figure(figsize=(10, 6))
    sns.regplot(data=male_df, x='Gestational_Week', y='Y_Concentration',
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Y-Chromosome Concentration vs. Gestational Week', fontsize=16)
    plt.xlabel('Gestational Week', fontsize=12)
    plt.ylabel('Y-Chromosome Concentration (%)', fontsize=12)
    plt.grid(True)
    plot1_path = os.path.join(plots_dir, 'p1_y_conc_vs_gest_week.png')
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved plot: {plot1_path}")

    # Plot 2: Y-Concentration vs. BMI
    plt.figure(figsize=(10, 6))
    sns.regplot(data=male_df, x='BMI', y='Y_Concentration',
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Y-Chromosome Concentration vs. Maternal BMI', fontsize=16)
    plt.xlabel('Maternal BMI', fontsize=12)
    plt.ylabel('Y-Chromosome Concentration (%)', fontsize=12)
    plt.grid(True)
    plot2_path = os.path.join(plots_dir, 'p1_y_conc_vs_bmi.png')
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved plot: {plot2_path}")

    # --- 2. Regression Modeling ---
    print("\nBuilding regression model...")

    # Prepare the data
    X = male_df[['Gestational_Week', 'BMI']]
    y = male_df['Y_Concentration']

    # Add a constant (for the intercept)
    X = sm.add_constant(X)

    # Build the model
    model = sm.OLS(y, X).fit()

    # Print the model summary
    print("\n--- Regression Model Summary ---")
    print(model.summary())

    # --- 3. Modeling with Interaction Term ---
    print("\nBuilding regression model with interaction term...")

    # Prepare the data with interaction term
    X_interact = male_df[['Gestational_Week', 'BMI']]
    X_interact['Gest_Week_x_BMI'] = X_interact['Gestational_Week'] * X_interact['BMI']
    X_interact = sm.add_constant(X_interact)

    # Build the model
    model_interact = sm.OLS(y, X_interact).fit()

    # Print the model summary
    print("\n--- Regression Model with Interaction Summary ---")
    print(model_interact.summary())

    print("\n--- Problem 1 Analysis Complete ---")

if __name__ == '__main__':
    # The script requires statsmodels and matplotlib.
    # You can install them with: pip install statsmodels matplotlib seaborn
    run_problem1_analysis()
