import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class Problem4_DynamicMultiStage:
    """
    Dynamic Multi-stage Detection Optimization Model for Female Fetuses
    Based on the dynamic programming formulation in problem4.md
    """
    
    def __init__(self, data_path="/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv"):
        """Initialize with processed data"""
        self.df = pd.read_csv(data_path)
        self.setup_data()
        self.T_max = 28  # Maximum gestational week
        self.T_min = 12  # Minimum gestational week
        self.max_detections = 3  # Maximum allowed detections
        
    def setup_data(self):
        """Setup and preprocess data for Problem 4 - Female fetuses analysis"""
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
        
        # Filter for potential female fetuses (lower Y concentration) with complete data
        # Use bottom 25% of Y concentrations as female proxy
        y_threshold = self.df['Y_concentration'].quantile(0.25)  # Bottom quartile
        
        self.female_data = self.df[
            (self.df['Y_concentration'] <= y_threshold) &  # Lower Y concentration (female proxy)
            (self.df['gestational_week'].notna()) &
            (self.df['gestational_week'] >= 12) &
            (self.df['gestational_week'] <= 28) &
            (self.df['孕妇BMI'].notna()) &
            (self.df['GC含量'].notna()) &
            (self.df['X染色体的Z值'].notna()) &
            (self.df['21号染色体的Z值'].notna()) &
            (self.df['18号染色体的Z值'].notna()) &
            (self.df['13号染色体的Z值'].notna())
        ].copy()
        
        # Use actual fetal health status as ground truth
        self.female_data['is_abnormal'] = (self.female_data['胎儿是否健康'] == '否')
        
        # Also create additional abnormality indicators based on chromosomal Z-values
        z_threshold = 2.0  # Standard threshold for abnormality
        
        self.female_data['chr21_abnormal'] = np.abs(self.female_data['21号染色体的Z值']) > z_threshold
        self.female_data['chr18_abnormal'] = np.abs(self.female_data['18号染色体的Z值']) > z_threshold
        self.female_data['chr13_abnormal'] = np.abs(self.female_data['13号染色体的Z值']) > z_threshold
        self.female_data['chrX_abnormal'] = np.abs(self.female_data['X染色体的Z值']) > z_threshold
        
        # Combined Z-score abnormality (any chromosome abnormal)
        self.female_data['z_abnormal'] = (
            self.female_data['chr21_abnormal'] | 
            self.female_data['chr18_abnormal'] | 
            self.female_data['chr13_abnormal'] | 
            self.female_data['chrX_abnormal']
        )
        
        if len(self.female_data) > 0:
            # Standardize features for state variables
            scaler = StandardScaler()
            features = ['孕妇BMI', 'GC含量', '21号染色体的Z值', '18号染色体的Z值', 
                       '13号染色体的Z值', 'X染色体的Z值']
            self.female_data[features] = scaler.fit_transform(self.female_data[features])
            
            # Calculate GC change rate (proxy)
            self.female_data = self.female_data.sort_values(['孕妇代码', 'gestational_week'])
            self.female_data['Delta_GC'] = self.female_data.groupby('孕妇代码')['GC含量'].diff().fillna(0)
        
        print(f"Female fetuses (proxy) for Problem 4 analysis: {len(self.female_data)}")
        print(f"Y concentration threshold (25th percentile): {y_threshold:.6f}")
        if len(self.female_data) > 0:
            print(f"Abnormality rate (actual health status): {self.female_data['is_abnormal'].mean()*100:.2f}%")
            print(f"Z-score abnormality rate: {self.female_data['z_abnormal'].mean()*100:.2f}%")
        
    def state_variables(self, t, bmi, gc_change, z_combined):
        """
        State space definition: s_t = (BMI_t, GC_t, Z_t, t)
        """
        return np.array([bmi, gc_change, z_combined, t])
    
    def sensitivity_function(self, state):
        """
        Sensitivity function: Sen(s_t) = 1/(1 + exp(-(0.8*Z_t + 0.2*ΔGC_t)))
        """
        bmi_t, gc_change, z_t, t = state
        return 1 / (1 + np.exp(-(0.8 * z_t + 0.2 * gc_change)))
    
    def risk_functions(self, t, detection_history):
        """
        Risk functions:
        Risk_1(t) = γ₁ * (t - 22)₊²  (late detection risk)
        Risk_2(t) = γ₂ * Σ(previous detections)  (repeated detection penalty)
        """
        gamma_1, gamma_2 = 0.1, 0.05
        
        # Late detection risk (t > 22 weeks)
        risk_1 = gamma_1 * max(0, t - 22) ** 2 if t > 22 else 0
        
        # Repeated detection penalty
        risk_2 = gamma_2 * sum(detection_history)
        
        return risk_1 + risk_2
    
    def reward_function(self, state, action, detection_history):
        """
        Reward function: R(s_t, a_t) = α·Sen(s_t)·a_t - β·[Risk_1(t) + Risk_2(t)]·a_t
        """
        alpha, beta = 0.7, 0.3
        bmi_t, gc_change, z_t, t = state
        
        if action == 1:  # Detection action
            sensitivity = self.sensitivity_function(state)
            risk = self.risk_functions(t, detection_history)
            return alpha * sensitivity - beta * risk
        else:  # No detection
            return 0
    
    def state_transition(self, current_state, action, noise_std=0.1):
        """
        State transition equations:
        BMI_{t+1} = BMI_t + ε_t^(BMI)
        ΔGC_{t+1} = 0.7 * ΔGC_t + η_t^(GC)
        """
        bmi_t, gc_change, z_t, t = current_state
        
        # BMI evolution with noise
        bmi_next = bmi_t + np.random.normal(0, noise_std)
        
        # GC change with autoregressive component
        gc_next = 0.7 * gc_change + np.random.normal(0, noise_std)
        
        # Z-score remains relatively stable (small random walk)
        z_next = z_t + np.random.normal(0, noise_std * 0.5)
        
        # Time increments
        t_next = t + 1
        
        return np.array([bmi_next, gc_next, z_next, t_next])
    
    def dynamic_programming_solver(self, sample_states):
        """
        Solve Bellman equation using backward induction:
        V_t(s_t) = max_{a_t} { R(s_t,a_t) + E[V_{t+1}(s_{t+1})] }
        """
        # Initialize value function and policy
        value_functions = {}
        optimal_policies = {}
        
        # Terminal condition (t = T_max)
        for state_idx in range(len(sample_states)):
            value_functions[(self.T_max, state_idx)] = 0
            optimal_policies[(self.T_max, state_idx)] = 0
        
        # Backward induction
        for t in range(self.T_max - 1, self.T_min - 1, -1):
            for state_idx, base_state in enumerate(sample_states):
                current_state = np.array([base_state[0], base_state[1], base_state[2], t])
                
                best_value = -np.inf
                best_action = 0
                
                # Try both actions (0: no detection, 1: detection)
                for action in [0, 1]:
                    # Skip if already reached max detections
                    detection_history = [0, 0, 0]  # Simplified for demonstration
                    if action == 1 and sum(detection_history) >= self.max_detections:
                        continue
                    
                    # Calculate immediate reward
                    immediate_reward = self.reward_function(current_state, action, detection_history)
                    
                    # Expected future value (simplified Monte Carlo approximation)
                    future_value = 0
                    n_simulations = 50
                    
                    for _ in range(n_simulations):
                        next_state = self.state_transition(current_state, action)
                        if t + 1 <= self.T_max:
                            # Approximate future value using nearest state
                            next_state_idx = min(len(sample_states) - 1, 
                                               max(0, int(np.linalg.norm(next_state[:3] - base_state[:3]) * 10) % len(sample_states)))
                            future_value += value_functions.get((t + 1, next_state_idx), 0)
                    
                    future_value /= n_simulations
                    
                    total_value = immediate_reward + future_value
                    
                    if total_value > best_value:
                        best_value = total_value
                        best_action = action
                
                value_functions[(t, state_idx)] = best_value
                optimal_policies[(t, state_idx)] = best_action
        
        return value_functions, optimal_policies
    
    def analyze_optimal_detection_strategy(self):
        """Analyze optimal detection strategy for female fetuses"""
        # Create representative states for analysis
        n_states = 100
        sample_states = []
        
        # Sample from actual data distribution
        for _ in range(n_states):
            idx = np.random.choice(len(self.female_data))
            row = self.female_data.iloc[idx]
            state = [row['孕妇BMI'], row['Delta_GC'], 
                    (row['21号染色体的Z值'] + row['18号染色体的Z值'] + 
                     row['13号染色体的Z值'] + row['X染色体的Z值']) / 4]
            sample_states.append(state)
        
        # Solve using dynamic programming
        print("Solving dynamic programming problem...")
        value_functions, optimal_policies = self.dynamic_programming_solver(sample_states)
        
        # Analyze results
        detection_recommendations = []
        
        for t in range(self.T_min, self.T_max):
            week_detections = []
            for state_idx in range(len(sample_states)):
                if (t, state_idx) in optimal_policies:
                    week_detections.append(optimal_policies[(t, state_idx)])
            
            if week_detections:
                detection_rate = np.mean(week_detections)
                detection_recommendations.append({
                    'week': t,
                    'detection_rate': detection_rate,
                    'recommended_action': 'Detect' if detection_rate > 0.5 else 'Wait'
                })
        
        return detection_recommendations, value_functions, optimal_policies
    
    def abnormality_classification(self):
        """
        Female fetal abnormality classification based on multiple factors
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.model_selection import train_test_split
        
        # Features for classification
        features = ['孕妇BMI', 'GC含量', 'X染色体的Z值', '21号染色体的Z值', 
                   '18号染色体的Z值', '13号染色体的Z值', 'gestational_week']
        
        X = self.female_data[features]
        y = self.female_data['is_abnormal']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                          random_state=42, stratify=y)
        
        # Train Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # Evaluation
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("Classification Results:")
        print(f"AUC Score: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return rf, feature_importance, auc
    
    def visualize_results(self, detection_recommendations, feature_importance):
        """Create comprehensive visualizations for Problem 4"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Problem 4: Dynamic Multi-stage Detection for Female Fetuses', fontsize=16)
        
        # Plot 1: Optimal detection strategy by week
        ax1 = axes[0, 0]
        weeks = [rec['week'] for rec in detection_recommendations]
        detection_rates = [rec['detection_rate'] for rec in detection_recommendations]
        
        bars = ax1.bar(weeks, detection_rates, color='lightcoral', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Gestational Week')
        ax1.set_ylabel('Detection Probability')
        ax1.set_title('Optimal Detection Strategy by Week')
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # Plot 2: Feature importance for abnormality detection
        ax2 = axes[0, 1]
        ax2.barh(feature_importance['Feature'], feature_importance['Importance'],
                color='lightblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Feature Importance for Abnormality Detection')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Chromosomal abnormality distribution
        ax3 = axes[1, 0]
        abnormality_types = ['Chr 21', 'Chr 18', 'Chr 13', 'Chr X']
        abnormality_rates = [
            self.female_data['chr21_abnormal'].mean(),
            self.female_data['chr18_abnormal'].mean(),
            self.female_data['chr13_abnormal'].mean(),
            self.female_data['chrX_abnormal'].mean()
        ]
        
        colors = ['red', 'orange', 'yellow', 'green']
        ax3.pie(abnormality_rates, labels=abnormality_types, colors=colors, 
                autopct='%1.1f%%', startangle=90)
        ax3.set_title('Distribution of Chromosomal Abnormalities')
        
        # Plot 4: Detection timing vs BMI
        ax4 = axes[1, 1]
        bmi_groups = pd.cut(self.female_data['孕妇BMI'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        week_groups = pd.cut(self.female_data['gestational_week'], bins=4, labels=['12-16w', '17-21w', '22-25w', '26-28w'])
        
        crosstab = pd.crosstab(bmi_groups, week_groups, normalize='index')
        im = ax4.imshow(crosstab.values, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(crosstab.columns)))
        ax4.set_xticklabels(crosstab.columns)
        ax4.set_yticks(range(len(crosstab.index)))
        ax4.set_yticklabels(crosstab.index)
        ax4.set_xlabel('Gestational Week Groups')
        ax4.set_ylabel('BMI Groups')
        ax4.set_title('Detection Timing vs BMI Distribution')
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, shrink=0.6)
        
        plt.tight_layout()
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/problem4_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self):
        """Run complete Problem 4 analysis"""
        print("=== Problem 4: Dynamic Multi-stage Detection for Female Fetuses ===")
        print()
        
        # Step 1: Analyze optimal detection strategy
        detection_recommendations, value_functions, optimal_policies = self.analyze_optimal_detection_strategy()
        
        print("Optimal Detection Strategy:")
        print("="*50)
        for rec in detection_recommendations[:10]:  # Show first 10 weeks
            print(f"Week {rec['week']}: Detection Rate = {rec['detection_rate']:.3f} ({rec['recommended_action']})")
        
        # Step 2: Abnormality classification
        classifier, feature_importance, auc = self.abnormality_classification()
        
        print(f"\nAbnormality Detection Performance:")
        print(f"AUC Score: {auc:.4f}")
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        # Step 3: Policy analysis
        optimal_weeks = []
        for rec in detection_recommendations:
            if rec['detection_rate'] > 0.5:
                optimal_weeks.append(rec['week'])
        
        if optimal_weeks:
            print(f"\nRecommended Detection Weeks: {optimal_weeks}")
            print(f"Primary Detection Window: {min(optimal_weeks)}-{max(optimal_weeks)} weeks")
        else:
            print("\nNo strong detection recommendations found in current analysis")
        
        # Step 4: Risk-benefit analysis
        total_detections = sum(1 for rec in detection_recommendations if rec['detection_rate'] > 0.5)
        print(f"\nRisk-Benefit Analysis:")
        print(f"Expected number of detections per patient: {total_detections/len(detection_recommendations)*len(detection_recommendations):.2f}")
        print(f"Detection efficiency (abnormalities detected / total detections): {self.female_data['is_abnormal'].mean():.3f}")
        
        # Step 5: Visualization
        self.visualize_results(detection_recommendations, feature_importance)
        
        # Save results
        results_df = pd.DataFrame(detection_recommendations)
        results_df.to_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem4_results.csv', index=False)
        
        feature_importance.to_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem4_feature_importance.csv', index=False)
        
        return {
            'detection_recommendations': detection_recommendations,
            'feature_importance': feature_importance,
            'auc_score': auc,
            'optimal_weeks': optimal_weeks
        }

if __name__ == "__main__":
    # Initialize and run Problem 4 analysis
    problem4 = Problem4_DynamicMultiStage()
    results = problem4.run_analysis()