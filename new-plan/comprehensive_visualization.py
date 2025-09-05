import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 12)
plt.style.use('seaborn-v0_8')

class ComprehensiveVisualization:
    """
    Create comprehensive visualizations for all problems with English labels
    """
    
    def __init__(self):
        """Initialize visualization class"""
        # Load results from all problems
        self.load_results()
        
    def load_results(self):
        """Load results from all problem analyses"""
        try:
            # Problem 2 results
            self.problem2_results = pd.read_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem2_results.csv')
            self.problem2_errors = pd.read_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem2_error_analysis.csv')
            
            # Problem 3 results
            self.problem3_results = pd.read_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem3_results.csv')
            
            # Problem 4 results
            self.problem4_results = pd.read_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem4_results.csv')
            self.problem4_features = pd.read_csv('/home/tfisher/code/math/2025/c-problem/new-plan/problem4_feature_importance.csv')
            
            # Original processed data
            self.processed_data = pd.read_csv('/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv')
            
            print("All results loaded successfully!")
            
        except Exception as e:
            print(f"Error loading results: {e}")
            print("Please ensure all problem analyses have been run first.")
    
    def create_overview_dashboard(self):
        """Create comprehensive dashboard overview"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('NIPT Analysis: Comprehensive Results Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # Row 1: Problem 2 Results
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_problem2_optimal_timing(ax1)
        
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_problem2_error_impact(ax2)
        
        # Row 2: Problem 3 Results
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_problem3_parameters(ax3)
        
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_problem3_validation(ax4)
        
        # Row 3: Problem 4 Results
        ax5 = fig.add_subplot(gs[2, :2])
        self.plot_problem4_detection_strategy(ax5)
        
        ax6 = fig.add_subplot(gs[2, 2:])
        self.plot_problem4_feature_importance(ax6)
        
        # Row 4: Overall Summary
        ax7 = fig.add_subplot(gs[3, :2])
        self.plot_overall_bmi_distribution(ax7)
        
        ax8 = fig.add_subplot(gs[3, 2:])
        self.plot_gestational_week_analysis(ax8)
        
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/comprehensive_dashboard.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_problem2_optimal_timing(self, ax):
        """Plot Problem 2 optimal timing results"""
        try:
            bars = ax.bar(range(len(self.problem2_results)), 
                         self.problem2_results['Optimal_Week'], 
                         color='steelblue', alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('BMI Groups')
            ax.set_ylabel('Optimal Detection Week')
            ax.set_title('Problem 2: Optimal NIPT Timing by BMI Group')
            ax.set_xticks(range(len(self.problem2_results)))
            ax.set_xticklabels([f"BMI {group}" for group in self.problem2_results['BMI_Group']], 
                              rotation=45, ha='right')
            ax.set_ylim(11, 29)
            
            # Add value labels
            for i, (bar, week) in enumerate(zip(bars, self.problem2_results['Optimal_Week'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{week:.1f}w', ha='center', va='bottom', fontweight='bold')
                
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Problem 2 data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_problem2_error_impact(self, ax):
        """Plot Problem 2 error impact analysis"""
        try:
            # Group by error weeks and calculate mean risk change
            error_impact = self.problem2_errors.groupby('Error_Weeks')['Risk_Change_Late'].mean()
            
            ax.plot(error_impact.index, error_impact.values, 'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Detection Error (weeks)')
            ax.set_ylabel('Average Risk Change')
            ax.set_title('Problem 2: Impact of Detection Errors')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(error_impact.index, error_impact.values, 1)
            p = np.poly1d(z)
            ax.plot(error_impact.index, p(error_impact.index), "r--", alpha=0.8, 
                   label=f'Trend: Risk increases {z[0]:.3f} per week error')
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Problem 2 error data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_problem3_parameters(self, ax):
        """Plot Problem 3 parameter comparison"""
        try:
            param_names = self.problem3_results['Parameter']
            original = self.problem3_results['Original_Value']
            updated = self.problem3_results['Updated_Value']
            uncertainty = self.problem3_results['Uncertainty']
            
            x = np.arange(len(param_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, original, width, label='Original Parameters', 
                          color='orange', alpha=0.7)
            bars2 = ax.bar(x + width/2, updated, width, label='Bayesian Updated', 
                          color='purple', alpha=0.7)
            
            # Add error bars for uncertainty
            ax.errorbar(x + width/2, updated, yerr=uncertainty, fmt='none', 
                       color='black', capsize=3)
            
            ax.set_xlabel('Model Parameters')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Problem 3: Parameter Update Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(param_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Problem 3 data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_problem3_validation(self, ax):
        """Plot Problem 3 validation metrics"""
        try:
            # Create mock validation results for visualization
            metrics = ['AUC Score', 'EARLY Score', 'ICI Score']
            values = [0.5912, 0.2461, 0.9417]  # From analysis output
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Score')
            ax.set_title('Problem 3: Cross-Validation Performance')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Problem 3 validation data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_problem4_detection_strategy(self, ax):
        """Plot Problem 4 detection strategy"""
        try:
            weeks = self.problem4_results['week']
            detection_rates = self.problem4_results['detection_rate']
            
            # Create color gradient based on detection rate
            colors = plt.cm.RdYlBu_r(detection_rates)
            
            bars = ax.bar(weeks, detection_rates, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Gestational Week')
            ax.set_ylabel('Detection Probability')
            ax.set_title('Problem 4: Dynamic Detection Strategy (Female Fetuses)')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, 
                      label='Decision Threshold (0.5)')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Problem 4 data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_problem4_feature_importance(self, ax):
        """Plot Problem 4 feature importance"""
        try:
            features = self.problem4_features['Feature'][:6]  # Top 6 features
            importance = self.problem4_features['Importance'][:6]
            
            bars = ax.barh(features, importance, color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Feature Importance')
            ax.set_title('Problem 4: Feature Importance for Abnormality Detection')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, imp in zip(bars, importance):
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                       f'{imp:.3f}', ha='left', va='center', fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Problem 4 feature data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_overall_bmi_distribution(self, ax):
        """Plot overall BMI distribution across all samples"""
        try:
            # Extract gestational week
            def extract_week(week_str):
                if pd.isna(week_str):
                    return np.nan
                try:
                    return float(str(week_str).split('w')[0])
                except:
                    return np.nan
            
            self.processed_data['gestational_week'] = self.processed_data['检测孕周'].apply(extract_week)
            
            # Create BMI distribution
            ax.hist(self.processed_data['孕妇BMI'].dropna(), bins=30, 
                   color='lightgreen', alpha=0.7, edgecolor='black', density=True)
            
            mean_bmi = self.processed_data['孕妇BMI'].mean()
            ax.axvline(x=mean_bmi, color='red', 
                      linestyle='--', linewidth=2, label=f'Mean BMI: {mean_bmi:.1f}')
            
            ax.set_xlabel('BMI')
            ax.set_ylabel('Density')
            ax.set_title('Overall BMI Distribution in Study Population')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'BMI data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_gestational_week_analysis(self, ax):
        """Plot gestational week distribution and Y concentration relationship"""
        try:
            # Extract gestational week
            def extract_week(week_str):
                if pd.isna(week_str):
                    return np.nan
                try:
                    return float(str(week_str).split('w')[0])
                except:
                    return np.nan
            
            self.processed_data['gestational_week'] = self.processed_data['检测孕周'].apply(extract_week)
            
            # Filter valid data
            valid_data = self.processed_data[
                (self.processed_data['gestational_week'].notna()) &
                (self.processed_data['gestational_week'] >= 10) &
                (self.processed_data['gestational_week'] <= 30)
            ]
            
            # Scatter plot of Y concentration vs gestational week
            scatter = ax.scatter(valid_data['gestational_week'], 
                               valid_data['Y_concentration'] * 100,  # Convert to percentage
                               c=valid_data['孕妇BMI'], cmap='viridis', 
                               alpha=0.6, s=30)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('BMI')
            
            # Add threshold line
            ax.axhline(y=4, color='red', linestyle='--', linewidth=2, 
                      label='4% Threshold')
            
            ax.set_xlabel('Gestational Week')
            ax.set_ylabel('Y Chromosome Concentration (%)')
            ax.set_title('Y Concentration vs Gestational Week (colored by BMI)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Gestational week data not available\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def create_individual_problem_summaries(self):
        """Create individual summary visualizations for each problem"""
        
        # Problem 2 Summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Problem 2 Summary: Time Window Constrained Optimization', fontsize=16)
        
        self.plot_problem2_optimal_timing(axes[0])
        self.plot_problem2_error_impact(axes[1])
        
        plt.tight_layout()
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/problem2_summary.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Problem 3 Summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Problem 3 Summary: Stratified Validation and Risk Quantification', fontsize=16)
        
        self.plot_problem3_parameters(axes[0])
        self.plot_problem3_validation(axes[1])
        
        plt.tight_layout()
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/problem3_summary.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Problem 4 Summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Problem 4 Summary: Dynamic Multi-stage Detection', fontsize=16)
        
        self.plot_problem4_detection_strategy(axes[0])
        self.plot_problem4_feature_importance(axes[1])
        
        plt.tight_layout()
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/problem4_summary.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_methodology_comparison(self):
        """Create a comparison of methodologies across all problems"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create methodology comparison table
        methods_data = {
            'Problem': ['Problem 2', 'Problem 3', 'Problem 4'],
            'Mathematical Model': ['Benefit Function with\\nSimulated Annealing', 
                                 'Cox Proportional\\nHazards Model', 
                                 'Dynamic Programming\\nwith Bellman Equation'],
            'Key Innovation': ['Time Window\\nConstraints', 
                              'Bayesian Parameter\\nUpdating', 
                              'Multi-stage\\nDecision Process'],
            'Target Population': ['Male Fetuses\\n(BMI Groups)', 
                                'Male Fetuses\\n(Risk Stratification)', 
                                'Female Fetuses\\n(Abnormality Detection)'],
            'Performance Metric': ['Risk Minimization', 'AUC & ICI', 'Detection Efficiency']
        }
        
        # Create a visual table
        table_data = []
        for i in range(len(methods_data['Problem'])):
            row = [methods_data[key][i] for key in methods_data.keys()]
            table_data.append(row)
        
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=list(methods_data.keys()),
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.25, 0.2, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(methods_data)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(methods_data)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        ax.set_title('Methodology Comparison Across All Problems', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig('/home/tfisher/code/math/2025/c-problem/new-plan/methodology_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_visualization_suite(self):
        """Run complete visualization suite"""
        print("Creating comprehensive visualizations...")
        print("="*50)
        
        # Main dashboard
        print("1. Creating overview dashboard...")
        self.create_overview_dashboard()
        
        # Individual summaries
        print("2. Creating individual problem summaries...")
        self.create_individual_problem_summaries()
        
        # Methodology comparison
        print("3. Creating methodology comparison...")
        self.create_methodology_comparison()
        
        print("\\nAll visualizations completed!")
        print("Files saved:")
        print("- comprehensive_dashboard.png")
        print("- problem2_summary.png")
        print("- problem3_summary.png") 
        print("- problem4_summary.png")
        print("- methodology_comparison.png")

if __name__ == "__main__":
    visualizer = ComprehensiveVisualization()
    visualizer.run_visualization_suite()