import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

from data_loader import get_cleaned_data

def run_problem4_analysis():
    """
    Performs the analysis for Problem 4: Female Fetus Anomaly Detection.
    """
    print("--- Running Problem 4 Analysis ---")

    # --- 0. Load Data & Setup ---
    try:
        _, female_df = get_cleaned_data()
        # Drop rows where critical features might be missing
        features = ['Age', 'BMI', 'Z_Score_13', 'Z_Score_18', 'Z_Score_21', 'Z_Score_X', 'X_Concentration', 'GC_Content']
        female_df.dropna(subset=features, inplace=True)
        print("Data loaded and prepared successfully.")
    except FileNotFoundError as e:
        print(e)
        return

    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # --- 1. Train a Classification Model (XGBoost) ---
    print("Training XGBClassifier model...")

    target = 'Is_Abnormal'

    X = female_df[features]
    y = female_df[target]

    # Split data, using stratification due to imbalanced classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the XGBoost Classifier
    # Use scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                             use_label_encoder=False, scale_pos_weight=scale_pos_weight, random_state=42)
    xgbc.fit(X_train, y_train)

    # --- 2. Evaluate Model Performance ---
    print("\n--- Model Evaluation ---")
    y_pred = xgbc.predict(X_test)
    y_pred_proba = xgbc.predict_proba(X_test)[:, 1]

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # AUC Score
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC Score: {roc_auc:.4f}")

    # --- 3. Generate and Save Plots ---
    # ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_curve_path = os.path.join(plots_dir, 'p4_roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"Saved ROC curve plot: {roc_curve_path}")

    # Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgbc, height=0.8)
    plt.title('XGBoost Feature Importance for Anomaly Detection', fontsize=16)
    plt.xlabel('F-score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    feature_imp_path = os.path.join(plots_dir, 'p4_feature_importance.png')
    plt.savefig(feature_imp_path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot: {feature_imp_path}")

    print("\n--- Problem 4 Analysis Complete ---")


if __name__ == '__main__':
    run_problem4_analysis()
