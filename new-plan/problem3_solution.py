import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class Problem3_StratifiedValidation:
    """
    Stratified Validation and Dynamic Risk Quantification Model
    Based on Cox Proportional Hazards Model extension from problem3.md
    """
    
    def __init__(self, data_path="/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv"):
        """Initialize with processed data"""
        self.df = pd.read_csv(data_path)
        self.setup_data()
        
    def setup_data(self):
        """Setup and preprocess data for Problem 3"""
        # Extract gestational week
        def extract_week(week_str):
            if pd.isna(week_str):
                return np.nan
            try:
                week_part = str(week_str).split('w')[0]
                return float(week_part)
            except:
                return np.nan
        
        self.df['gestational_week'] = self.df['检测孕周'].apply(extract_week)
        
        # Filter for male fetuses with complete data
        self.male_data = self.df[
            (self.df['Y_concentration'] > 0) & 
            (self.df['gestational_week'].notna()) &
            (self.df['gestational_week'] >= 12) &
            (self.df['gestational_week'] <= 28) &
            (self.df['孕妇BMI'].notna()) &
            (self.df['GC含量'].notna())
        ].copy()
        
        # Create time-dependent covariates
        self.male_data['BMI_t'] = (self.male_data['孕妇BMI'] - self.male_data['孕妇BMI'].mean()) / self.male_data['孕妇BMI'].std()
        
        # Calculate GC change (Delta GC)
        self.male_data = self.male_data.sort_values(['孕妇代码', 'gestational_week'])
        self.male_data['Delta_GC'] = self.male_data.groupby('孕妇代码')['GC含量'].diff().fillna(0)
        
        # Z-score for current gestational week (using Y chromosome Z-value)
        self.male_data['Z_t'] = (self.male_data['Y染色体的Z值'] - self.male_data['Y染色体的Z值'].mean()) / self.male_data['Y染色体的Z值'].std()
        
        # Define events (abnormal detection)
        self.male_data['Y_above_threshold'] = self.male_data['Y_concentration'] >= 0.04
        self.male_data['event'] = (~self.male_data['Y_above_threshold']).astype(int)  # Event = failure to reach threshold
        
        print(f"Male fetuses for Problem 3 analysis: {len(self.male_data)}")
        print(f"Event rate (failure to reach threshold): {self.male_data['event'].mean()*100:.2f}%")
        
    def cox_proportional_hazards_approximation(self):
        """
        Approximate Cox proportional hazards model using logistic regression
        h(t|X) = h_0(t) * exp[β₁·BMI(t) + β₂·ΔGC(t) + β₃·Z(t)]
        """
        # Prepare features for the Cox model approximation
        X = self.male_data[['BMI_t', 'Delta_GC', 'Z_t']].fillna(0)
        y = self.male_data['event']
        
        # Fit logistic regression as approximation to Cox model
        cox_model = LogisticRegression(random_state=42, max_iter=1000)
        cox_model.fit(X, y)
        
        # Extract coefficients (beta parameters)
        beta_1, beta_2, beta_3 = cox_model.coef_[0]
        intercept = cox_model.intercept_[0]
        
        # Calculate survival function approximation
        risk_scores = cox_model.predict_proba(X)[:, 1]
        
        print(f"Cox Model Coefficients:")
        print(f"β₁ (BMI effect): {beta_1:.4f}")
        print(f"β₂ (ΔGC effect): {beta_2:.4f}")
        print(f"β₃ (Z-score effect): {beta_3:.4f}")
        print(f"Intercept: {intercept:.4f}")
        
        return cox_model, (beta_1, beta_2, beta_3, intercept), risk_scores
    
    def optimal_detection_time_decision(self, cox_model, beta_params):
        """
        Optimal detection time decision function:
        max_T [S(T) - λ·Risk(T)]
        """
        beta_1, beta_2, beta_3, intercept = beta_params
        lambda_penalty = 0.3  # Risk penalty factor
        
        optimal_times = []
        
        for _, row in self.male_data.iterrows():
            bmi_t = row['BMI_t']
            delta_gc = row['Delta_GC']
            z_t = row['Z_t']
            
            def dual_objective(T):
                # Survival function approximation: S(T) = exp(-∫h(u)du)
                # Simplified as: S(T) = 1 / (1 + exp(β₀ + β₁·BMI + β₂·ΔGC + β₃·Z))
                survival_prob = 1 / (1 + np.exp(intercept + beta_1*bmi_t + beta_2*delta_gc + beta_3*z_t))
                
                # Risk penalty function
                risk_penalty = 0
                if T < 12:
                    risk_penalty = 0.5  # γ₁ for early detection
                elif 12 <= T <= 22:
                    risk_penalty = 0.1 * (T - 16)**2  # γ₂ for mid-term
                elif T > 22:
                    risk_penalty = 1.0  # γ₃ for late detection
                
                return survival_prob - lambda_penalty * risk_penalty
            
            # Find optimal time in [12, 28] range
            T_range = np.linspace(12, 28, 100)
            objectives = [dual_objective(T) for T in T_range]
            optimal_T = T_range[np.argmax(objectives)]
            
            optimal_times.append(optimal_T)
        
        self.male_data['optimal_detection_time'] = optimal_times
        
        return optimal_times
    
    def stratified_cross_validation(self, cox_model):
        """
        Stratified cross-validation with Inter-group Consistency Index (ICI)
        """
        X = self.male_data[['BMI_t', 'Delta_GC', 'Z_t']].fillna(0)
        y = self.male_data['event']
        
        # Create BMI-based stratification
        bmi_quartiles = pd.qcut(self.male_data['孕妇BMI'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Perform stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        group_aucs = []
        group_early_scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model on training set
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Calculate AUC
            auc = roc_auc_score(y_test, y_pred_proba)
            group_aucs.append(auc)
            
            # Calculate EARLY score (early detection effectiveness)
            test_weeks = self.male_data.iloc[test_idx]['gestational_week']
            optimal_weeks = self.male_data.iloc[test_idx]['optimal_detection_time']
            
            early_score = np.mean(
                (optimal_weeks < test_weeks) * np.exp(-0.1 * (test_weeks - optimal_weeks))
            )
            group_early_scores.append(early_score)
        
        # Calculate pooled metrics
        pooled_auc = np.mean(group_aucs)
        pooled_early = np.mean(group_early_scores)
        
        # Calculate Inter-group Consistency Index (ICI)
        auc_deviations = np.abs(np.array(group_aucs) - pooled_auc)
        early_deviations = np.abs(np.array(group_early_scores) - pooled_early)
        
        ici = 1 - np.mean((auc_deviations + early_deviations) / (pooled_auc + pooled_early))
        
        print(f"\nStratified Cross-Validation Results:")
        print(f"Pooled AUC: {pooled_auc:.4f}")
        print(f"Pooled EARLY score: {pooled_early:.4f}")
        print(f"Inter-group Consistency Index (ICI): {ici:.4f}")
        
        return {
            'group_aucs': group_aucs,
            'group_early_scores': group_early_scores,
            'pooled_auc': pooled_auc,
            'pooled_early': pooled_early,
            'ici': ici
        }
    
    def bayesian_parameter_update(self, beta_params, n_bootstrap=50):
        """
        Bayesian hierarchical model for parameter updating
        β^(m+1) ~ N(μ^(m), Σ^(m))
        """
        beta_1, beta_2, beta_3, intercept = beta_params
        alpha = 0.9  # Memory factor
        
        # Bootstrap sampling for parameter uncertainty
        bootstrap_betas = []
        
        X = self.male_data[['BMI_t', 'Delta_GC', 'Z_t']].fillna(0)
        y = self.male_data['event']
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X)
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[boot_indices]
            y_boot = y.iloc[boot_indices]
            
            # Fit model on bootstrap sample
            model = LogisticRegression(random_state=i, max_iter=1000)
            try:
                model.fit(X_boot, y_boot)
                bootstrap_betas.append(list(model.coef_[0]) + [model.intercept_[0]])
            except:
                continue
        
        bootstrap_betas = np.array(bootstrap_betas)
        
        # Update parameters using Bayesian update rule
        current_params = np.array([beta_1, beta_2, beta_3, intercept])
        bootstrap_mean = np.mean(bootstrap_betas, axis=0)
        
        updated_params = alpha * current_params + (1 - alpha) * bootstrap_mean
        param_uncertainty = np.std(bootstrap_betas, axis=0)
        
        print(f"\nBayesian Parameter Update:")
        print(f"Updated β₁: {updated_params[0]:.4f} ± {param_uncertainty[0]:.4f}")
        print(f"Updated β₂: {updated_params[1]:.4f} ± {param_uncertainty[1]:.4f}")
        print(f"Updated β₃: {updated_params[2]:.4f} ± {param_uncertainty[2]:.4f}")
        print(f"Updated intercept: {updated_params[3]:.4f} ± {param_uncertainty[3]:.4f}")
        
        return updated_params, param_uncertainty
    
    def visualize_results(self, cv_results, beta_params, updated_params):
        """Create comprehensive visualizations for Problem 3"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Problem 3: Stratified Validation and Dynamic Risk Quantification', fontsize=16)
        
        # Plot 1: Cross-validation AUC scores
        ax1 = axes[0, 0]
        ax1.bar(range(len(cv_results['group_aucs'])), cv_results['group_aucs'], 
                color='lightblue', alpha=0.7, edgecolor='black')
        ax1.axhline(y=cv_results['pooled_auc'], color='red', linestyle='--', 
                   label=f'Pooled AUC: {cv_results["pooled_auc"]:.3f}')
        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Cross-Validation AUC Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: EARLY scores across folds
        ax2 = axes[0, 1]
        ax2.bar(range(len(cv_results['group_early_scores'])), cv_results['group_early_scores'],
                color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.axhline(y=cv_results['pooled_early'], color='red', linestyle='--',
                   label=f'Pooled EARLY: {cv_results["pooled_early"]:.3f}')
        ax2.set_xlabel('Fold Number')
        ax2.set_ylabel('EARLY Score')
        ax2.set_title('Early Detection Effectiveness Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameter comparison (before and after Bayesian update)
        ax3 = axes[1, 0]
        param_names = ['β₁ (BMI)', 'β₂ (ΔGC)', 'β₃ (Z)', 'Intercept']
        original = beta_params
        updated = updated_params[:4]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        ax3.bar(x - width/2, original, width, label='Original', color='orange', alpha=0.7)
        ax3.bar(x + width/2, updated, width, label='Bayesian Updated', color='purple', alpha=0.7)
        
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Parameter Value')
        ax3.set_title('Parameter Update Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(param_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Optimal detection time distribution
        ax4 = axes[1, 1]
        ax4.hist(self.male_data['optimal_detection_time'], bins=20, 
                color='skyblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=self.male_data['optimal_detection_time'].mean(), 
                   color='red', linestyle='--', 
                   label=f'Mean: {self.male_data["optimal_detection_time"].mean():.1f} weeks')
        ax4.set_xlabel('Optimal Detection Time (weeks)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Optimal Detection Times')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/problem3_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self):
        """Run complete Problem 3 analysis"""
        print("=== Problem 3: Stratified Validation and Dynamic Risk Quantification ===")
        print()
        
        # Step 1: Cox proportional hazards approximation
        cox_model, beta_params, risk_scores = self.cox_proportional_hazards_approximation()
        
        # Step 2: Optimal detection time decision
        optimal_times = self.optimal_detection_time_decision(cox_model, beta_params)
        
        print(f"\nOptimal Detection Time Statistics:")
        print(f"Mean optimal time: {np.mean(optimal_times):.2f} weeks")
        print(f"Std optimal time: {np.std(optimal_times):.2f} weeks")
        print(f"Range: {np.min(optimal_times):.1f} - {np.max(optimal_times):.1f} weeks")
        
        # Step 3: Stratified cross-validation
        cv_results = self.stratified_cross_validation(cox_model)
        
        # Step 4: Bayesian parameter updating
        updated_params, param_uncertainty = self.bayesian_parameter_update(beta_params)
        
        # Step 5: Visualization
        self.visualize_results(cv_results, beta_params, updated_params)
        
        # Save results
        results_summary = {
            'original_beta_params': beta_params,
            'updated_beta_params': updated_params.tolist(),
            'parameter_uncertainty': param_uncertainty.tolist(),
            'cv_results': cv_results,
            'optimal_time_stats': {
                'mean': np.mean(optimal_times),
                'std': np.std(optimal_times),
                'min': np.min(optimal_times),
                'max': np.max(optimal_times)
            }
        }
        
        # Save to CSV
        results_df = pd.DataFrame({
            'Parameter': ['β₁ (BMI)', 'β₂ (ΔGC)', 'β₃ (Z)', 'Intercept'],
            'Original_Value': beta_params,
            'Updated_Value': updated_params[:4],
            'Uncertainty': param_uncertainty[:4]
        })
        results_df.to_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem3_results.csv', index=False)
        
        return results_summary

if __name__ == "__main__":
    # Initialize and run Problem 3 analysis
    problem3 = Problem3_StratifiedValidation()
    results = problem3.run_analysis()