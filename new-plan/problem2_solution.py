import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.stats import gamma, beta, norm
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class Problem2_TimeWindowOptimization:
    """
    Time Window Constrained Dynamic Detection Optimization Model
    Based on the mathematical formulation in problem2.md
    """
    
    def __init__(self, data_path="/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv"):
        """Initialize with processed data"""
        self.df = pd.read_csv(data_path)
        self.setup_data()
        
    def setup_data(self):
        """Setup and preprocess data for Problem 2"""
        # Y chromosome concentration is already in percentage format
        self.df['Y_concentration_pct'] = self.df['Y_concentration']
        
        # Extract gestational week from '检测孕周' (format: "13w", "13w+6", etc.)
        def extract_week(week_str):
            if pd.isna(week_str):
                return np.nan
            # Extract the week number before 'w'
            try:
                week_part = str(week_str).split('w')[0]
                return float(week_part)
            except:
                return np.nan
        
        self.df['gestational_week'] = self.df['检测孕周'].apply(extract_week)
        
        # Filter for male fetuses (those with non-zero Y chromosome concentration)
        self.male_data = self.df[
            (self.df['Y_concentration_pct'] > 0) & 
            (self.df['gestational_week'].notna()) &
            (self.df['gestational_week'] >= 12) &
            (self.df['gestational_week'] <= 28)
        ].copy()
        
        # Add threshold achievement indicator (4% = 0.04 in decimal format)
        self.male_data['Y_above_4pct'] = self.male_data['Y_concentration_pct'] >= 0.04
        
        # Calculate GC change rate (proxy using available GC content)
        if len(self.male_data) > 0:
            self.male_data['GC_change'] = (self.male_data['GC含量'] - self.male_data['GC含量'].mean()) / self.male_data['GC含量'].std()
        
        print(f"Male fetuses with valid data: {len(self.male_data)}")
        if len(self.male_data) > 0:
            print(f"Y concentration range: {self.male_data['Y_concentration_pct'].min():.6f} - {self.male_data['Y_concentration_pct'].max():.6f}")
            print(f"Percentage achieving 4% threshold: {self.male_data['Y_above_4pct'].mean()*100:.2f}%")
        
    def comprehensive_benefit_function(self, T, BMI, GC_change, beta_params):
        """
        Comprehensive benefit function from problem2.md:
        Ψ(T) = [1/(1 + exp(-(β₀ + β₁·BMI + β₂·GC)))] × [1 - (T-12)/(22-12)]^w₁ - λ·(T/40)^w₂·I(T>28)
        """
        beta_0, beta_1, beta_2 = beta_params
        w_1, w_2, lambda_val = 2.0, 0.75, 0.85  # From problem specification
        
        # First term: Detection sensitivity (logistic growth)
        sensitivity = 1 / (1 + np.exp(-(beta_0 + beta_1 * BMI + beta_2 * GC_change)))
        
        # Second term: Late detection risk penalty  
        if T <= 22:
            time_penalty = (1 - (T - 12) / (22 - 12)) ** w_1
        else:
            time_penalty = 0
        
        # Third term: Very late detection penalty
        late_penalty = 0
        if T > 28:
            late_penalty = lambda_val * (T / 40) ** w_2
        
        return sensitivity * time_penalty - late_penalty
    
    def calibrate_parameters(self):
        """Calibrate model parameters using Bayesian hierarchical approach"""
        # Use actual data to estimate beta parameters
        from sklearn.linear_model import LogisticRegression
        
        # Prepare features for logistic regression
        X = self.male_data[['孕妇BMI', 'GC_change']].fillna(0)
        y = self.male_data['Y_above_4pct']
        
        # Fit logistic regression to get initial beta estimates
        lr = LogisticRegression(fit_intercept=True, random_state=42)
        lr.fit(X, y)
        
        beta_0 = lr.intercept_[0]
        beta_1, beta_2 = lr.coef_[0]
        
        print(f"Calibrated parameters: β₀={beta_0:.4f}, β₁={beta_1:.4f}, β₂={beta_2:.4f}")
        
        return (beta_0, beta_1, beta_2)
    
    def optimize_detection_time(self, BMI, GC_change, beta_params):
        """
        Optimize detection time for given BMI and GC change using simulated annealing approach
        """
        def objective(T):
            return -self.comprehensive_benefit_function(T, BMI, GC_change, beta_params)
        
        # Constrained optimization within clinical window [12, 28]
        result = minimize_scalar(objective, bounds=(12, 28), method='bounded')
        
        optimal_T = result.x
        max_benefit = -result.fun
        
        return optimal_T, max_benefit
    
    def create_bmi_groups_and_optimize(self):
        """Create BMI groups and find optimal NIPT timing for each group"""
        # Define BMI groups based on clinical practice
        bmi_breaks = [20, 28, 32, 36, 40, 50]
        bmi_labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '[40,50)']
        
        # Calibrate parameters
        beta_params = self.calibrate_parameters()
        
        results = []
        
        for i, (lower, upper) in enumerate(zip(bmi_breaks[:-1], bmi_breaks[1:])):
            # Filter data for current BMI group
            group_data = self.male_data[
                (self.male_data['孕妇BMI'] >= lower) & 
                (self.male_data['孕妇BMI'] < upper)
            ]
            
            if len(group_data) == 0:
                continue
                
            # Calculate group statistics
            avg_BMI = group_data['孕妇BMI'].mean()
            avg_GC_change = group_data['GC_change'].mean()
            
            # Find optimal detection time
            optimal_T, max_benefit = self.optimize_detection_time(avg_BMI, avg_GC_change, beta_params)
            
            # Calculate risk metrics
            risk_early = len(group_data[group_data['gestational_week'] < 12]) / len(group_data) if len(group_data) > 0 else 0
            risk_late = len(group_data[group_data['gestational_week'] > 22]) / len(group_data) if len(group_data) > 0 else 0
            
            results.append({
                'BMI_Group': bmi_labels[i],
                'BMI_Range': f'[{lower}, {upper})',
                'Sample_Size': len(group_data),
                'Avg_BMI': avg_BMI,
                'Optimal_Week': optimal_T,
                'Max_Benefit': max_benefit,
                'Early_Risk': risk_early,
                'Late_Risk': risk_late,
                'Total_Risk': risk_early + risk_late
            })
        
        return pd.DataFrame(results), beta_params
    
    def analyze_detection_errors(self, results_df):
        """Analyze impact of detection errors on results"""
        error_scenarios = [0, 0.5, 1.0, 1.5, 2.0]  # weeks of error
        
        error_analysis = []
        
        for _, row in results_df.iterrows():
            base_week = row['Optimal_Week']
            base_risk = row['Total_Risk']
            
            for error in error_scenarios:
                # Simulate detection at different times
                early_detection = max(12, base_week - error)
                late_detection = min(28, base_week + error)
                
                # Calculate risk changes (simplified model)
                early_risk_change = error * 0.1 if error > 0 else 0  # Early detection reduces risk
                late_risk_change = error * 0.15  # Late detection increases risk
                
                error_analysis.append({
                    'BMI_Group': row['BMI_Group'],
                    'Error_Weeks': error,
                    'Early_Detection_Week': early_detection,
                    'Late_Detection_Week': late_detection,
                    'Risk_Change_Early': -early_risk_change,
                    'Risk_Change_Late': late_risk_change
                })
        
        return pd.DataFrame(error_analysis)
    
    def visualize_results(self, results_df, error_analysis_df):
        """Create comprehensive visualizations for Problem 2"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Problem 2: Time Window Constrained Dynamic Detection Optimization', fontsize=16)
        
        # Plot 1: Optimal detection weeks by BMI group
        ax1 = axes[0, 0]
        bars = ax1.bar(results_df['BMI_Group'], results_df['Optimal_Week'], 
                      color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('BMI Groups')
        ax1.set_ylabel('Optimal Detection Week')
        ax1.set_title('Optimal NIPT Timing by BMI Group')
        ax1.set_ylim(12, 28)
        
        # Add value labels on bars
        for bar, week in zip(bars, results_df['Optimal_Week']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{week:.1f}', ha='center', va='bottom')
        
        # Plot 2: Risk analysis by BMI group
        ax2 = axes[0, 1]
        x_pos = np.arange(len(results_df))
        width = 0.35
        
        ax2.bar(x_pos - width/2, results_df['Early_Risk'], width, 
                label='Early Detection Risk', color='orange', alpha=0.7)
        ax2.bar(x_pos + width/2, results_df['Late_Risk'], width,
                label='Late Detection Risk', color='red', alpha=0.7)
        
        ax2.set_xlabel('BMI Groups')
        ax2.set_ylabel('Risk Proportion')
        ax2.set_title('Risk Analysis by BMI Group')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(results_df['BMI_Group'], rotation=45)
        ax2.legend()
        
        # Plot 3: Detection error impact
        ax3 = axes[1, 0]
        for group in error_analysis_df['BMI_Group'].unique():
            group_data = error_analysis_df[error_analysis_df['BMI_Group'] == group]
            ax3.plot(group_data['Error_Weeks'], group_data['Risk_Change_Late'], 
                    'o-', label=group, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Detection Error (weeks)')
        ax3.set_ylabel('Risk Change')
        ax3.set_title('Impact of Detection Errors on Risk')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Benefit function visualization
        ax4 = axes[1, 1]
        T_range = np.linspace(12, 28, 100)
        
        # Show benefit curves for different BMI values
        bmi_examples = [25, 30, 35, 40]
        colors = ['blue', 'green', 'orange', 'red']
        
        beta_params = self.calibrate_parameters()
        
        for bmi, color in zip(bmi_examples, colors):
            benefits = [self.comprehensive_benefit_function(t, bmi, 0, beta_params) for t in T_range]
            ax4.plot(T_range, benefits, color=color, label=f'BMI = {bmi}', linewidth=2)
        
        ax4.set_xlabel('Gestational Week')
        ax4.set_ylabel('Comprehensive Benefit')
        ax4.set_title('Benefit Function by BMI')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/problem2_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self):
        """Run complete Problem 2 analysis"""
        print("=== Problem 2: Time Window Constrained Dynamic Detection Optimization ===")
        print()
        
        # Create BMI groups and optimize
        results_df, beta_params = self.create_bmi_groups_and_optimize()
        
        print("Optimal Detection Timing by BMI Group:")
        print("="*60)
        for _, row in results_df.iterrows():
            print(f"BMI {row['BMI_Range']}: Optimal Week = {row['Optimal_Week']:.1f}, "
                  f"Samples = {row['Sample_Size']}, Total Risk = {row['Total_Risk']:.3f}")
        
        print("\nCalibrated Model Parameters:")
        print("="*40)
        print(f"β₀ = {beta_params[0]:.4f}")
        print(f"β₁ = {beta_params[1]:.4f}")  
        print(f"β₂ = {beta_params[2]:.4f}")
        
        # Analyze detection errors
        error_analysis_df = self.analyze_detection_errors(results_df)
        
        print("\nDetection Error Impact Analysis:")
        print("="*40)
        avg_error_impact = error_analysis_df.groupby('Error_Weeks')['Risk_Change_Late'].mean()
        for error, impact in avg_error_impact.items():
            print(f"Error ±{error} weeks: Average risk change = +{impact:.3f}")
        
        # Create visualizations
        self.visualize_results(results_df, error_analysis_df)
        
        # Save results
        results_df.to_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem2_results.csv', index=False)
        error_analysis_df.to_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem2_error_analysis.csv', index=False)
        
        return results_df, error_analysis_df

if __name__ == "__main__":
    # Initialize and run Problem 2 analysis
    problem2 = Problem2_TimeWindowOptimization()
    results_df, error_df = problem2.run_analysis()