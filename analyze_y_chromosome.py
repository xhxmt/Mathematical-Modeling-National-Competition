import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Read the Excel file
file_path = '/home/tfisher/code/math/2025/c-problem/附件.xlsx'

# Load male fetus data (contains Y chromosome concentration data)
df_male = pd.read_excel(file_path, sheet_name='男胎检测数据')

print(f"Male fetal data shape: {df_male.shape}")
print("\nBasic statistics of key variables:")
print("="*50)

# Key variables for analysis
key_vars = ['检测孕周', '孕妇BMI', 'Y染色体浓度']

# Check data types first
print("Data types:")
for var in key_vars:
    if var in df_male.columns:
        print(f"{var}: {df_male[var].dtype}")
        print(f"Sample values: {df_male[var].head().tolist()}")
        print()

# Clean and convert data to numeric
def parse_gestational_week(gw_str):
    """Parse gestational week string like '11w+6' to numeric weeks"""
    if pd.isna(gw_str) or gw_str == '':
        return np.nan
    
    try:
        gw_str = str(gw_str)
        if 'w' in gw_str:
            parts = gw_str.replace('w+', 'w ').replace('w', '').strip()
            if ' ' in parts:
                weeks, days = parts.split(' ', 1)
                return float(weeks) + float(days.strip()) / 7.0
            else:
                return float(parts)
        else:
            return float(gw_str)
    except:
        return np.nan

# Parse gestational week data
df_male['检测孕周_numeric'] = df_male['检测孕周'].apply(parse_gestational_week)

# Update key variables
key_vars = ['检测孕周_numeric', '孕妇BMI', 'Y染色体浓度']

# Now show statistics
for var in key_vars:
    if var in df_male.columns:
        print(f"\n{var}:")
        print(f"  Count: {df_male[var].notna().sum()}")
        if df_male[var].notna().sum() > 0:
            print(f"  Mean: {df_male[var].mean():.4f}")
            print(f"  Std: {df_male[var].std():.4f}")
            print(f"  Min: {df_male[var].min():.4f}")
            print(f"  Max: {df_male[var].max():.4f}")
        else:
            print("  No valid numeric data found")

# Remove missing values
df_clean = df_male[key_vars].dropna()
print(f"\nClean data shape after removing missing values: {df_clean.shape}")

# Extract variables
gestational_week = df_clean['检测孕周_numeric'].values
bmi = df_clean['孕妇BMI'].values  
y_concentration = df_clean['Y染色体浓度'].values

print("\n" + "="*60)
print("ANALYSIS 1: Y chromosome concentration vs Gestational Week")
print("="*60)

# 1. Linear regression model
X_gw = gestational_week.reshape(-1, 1)
y_target = y_concentration

# Linear model
linear_model_gw = LinearRegression()
linear_model_gw.fit(X_gw, y_target)
y_pred_linear_gw = linear_model_gw.predict(X_gw)
r2_linear_gw = r2_score(y_target, y_pred_linear_gw)

# Polynomial models (2nd and 3rd degree)
poly2_features_gw = PolynomialFeatures(degree=2)
X_poly2_gw = poly2_features_gw.fit_transform(X_gw)
poly2_model_gw = LinearRegression()
poly2_model_gw.fit(X_poly2_gw, y_target)
y_pred_poly2_gw = poly2_model_gw.predict(X_poly2_gw)
r2_poly2_gw = r2_score(y_target, y_pred_poly2_gw)

poly3_features_gw = PolynomialFeatures(degree=3)
X_poly3_gw = poly3_features_gw.fit_transform(X_gw)
poly3_model_gw = LinearRegression()
poly3_model_gw.fit(X_poly3_gw, y_target)
y_pred_poly3_gw = poly3_model_gw.predict(X_poly3_gw)
r2_poly3_gw = r2_score(y_target, y_pred_poly3_gw)

# Correlation analysis
corr_gw, p_value_gw = stats.pearsonr(gestational_week, y_concentration)

print(f"Correlation coefficient (Pearson): {corr_gw:.4f}")
print(f"P-value: {p_value_gw:.4e}")
print(f"Linear model R²: {r2_linear_gw:.4f}")
print(f"Polynomial (degree 2) R²: {r2_poly2_gw:.4f}")
print(f"Polynomial (degree 3) R²: {r2_poly3_gw:.4f}")

# Print the best model equations
print(f"\nLinear model equation:")
print(f"Y_concentration = {linear_model_gw.coef_[0]:.6f} × GestationalWeek + {linear_model_gw.intercept_:.6f}")

print(f"\nPolynomial (degree 2) model coefficients:")
coeffs_poly2_gw = poly2_model_gw.coef_
print(f"Y_concentration = {coeffs_poly2_gw[2]:.6e} × GW² + {coeffs_poly2_gw[1]:.6f} × GW + {coeffs_poly2_gw[0]:.6f}")

print(f"\nPolynomial (degree 3) model coefficients:")
coeffs_poly3_gw = poly3_model_gw.coef_
print(f"Y_concentration = {coeffs_poly3_gw[3]:.6e} × GW³ + {coeffs_poly3_gw[2]:.6e} × GW² + {coeffs_poly3_gw[1]:.6f} × GW + {coeffs_poly3_gw[0]:.6f}")

print("\n" + "="*60)
print("ANALYSIS 2: Y chromosome concentration vs BMI")
print("="*60)

# 2. BMI analysis
X_bmi = bmi.reshape(-1, 1)

# Linear model
linear_model_bmi = LinearRegression()
linear_model_bmi.fit(X_bmi, y_target)
y_pred_linear_bmi = linear_model_bmi.predict(X_bmi)
r2_linear_bmi = r2_score(y_target, y_pred_linear_bmi)

# Polynomial models (2nd and 3rd degree)
poly2_features_bmi = PolynomialFeatures(degree=2)
X_poly2_bmi = poly2_features_bmi.fit_transform(X_bmi)
poly2_model_bmi = LinearRegression()
poly2_model_bmi.fit(X_poly2_bmi, y_target)
y_pred_poly2_bmi = poly2_model_bmi.predict(X_poly2_bmi)
r2_poly2_bmi = r2_score(y_target, y_pred_poly2_bmi)

poly3_features_bmi = PolynomialFeatures(degree=3)
X_poly3_bmi = poly3_features_bmi.fit_transform(X_bmi)
poly3_model_bmi = LinearRegression()
poly3_model_bmi.fit(X_poly3_bmi, y_target)
y_pred_poly3_bmi = poly3_model_bmi.predict(X_poly3_bmi)
r2_poly3_bmi = r2_score(y_target, y_pred_poly3_bmi)

# Correlation analysis
corr_bmi, p_value_bmi = stats.pearsonr(bmi, y_concentration)

print(f"Correlation coefficient (Pearson): {corr_bmi:.4f}")
print(f"P-value: {p_value_bmi:.4e}")
print(f"Linear model R²: {r2_linear_bmi:.4f}")
print(f"Polynomial (degree 2) R²: {r2_poly2_bmi:.4f}")
print(f"Polynomial (degree 3) R²: {r2_poly3_bmi:.4f}")

# Print the best model equations
print(f"\nLinear model equation:")
print(f"Y_concentration = {linear_model_bmi.coef_[0]:.6f} × BMI + {linear_model_bmi.intercept_:.6f}")

print(f"\nPolynomial (degree 2) model coefficients:")
coeffs_poly2_bmi = poly2_model_bmi.coef_
print(f"Y_concentration = {coeffs_poly2_bmi[2]:.6e} × BMI² + {coeffs_poly2_bmi[1]:.6f} × BMI + {coeffs_poly2_bmi[0]:.6f}")

print(f"\nPolynomial (degree 3) model coefficients:")
coeffs_poly3_bmi = poly3_model_bmi.coef_
print(f"Y_concentration = {coeffs_poly3_bmi[3]:.6e} × BMI³ + {coeffs_poly3_bmi[2]:.6e} × BMI² + {coeffs_poly3_bmi[1]:.6f} × BMI + {coeffs_poly3_bmi[0]:.6f}")

print("\n" + "="*60)
print("SIGNIFICANCE TESTS")
print("="*60)

# F-test for model significance
from scipy.stats import f

def f_test_significance(y_true, y_pred, n_params):
    """Perform F-test for model significance"""
    n = len(y_true)
    ssr = np.sum((y_pred - np.mean(y_true))**2)  # Sum of squares regression
    sse = np.sum((y_true - y_pred)**2)  # Sum of squares error
    msr = ssr / (n_params - 1)  # Mean square regression
    mse = sse / (n - n_params)  # Mean square error
    f_stat = msr / mse
    p_value = 1 - f.cdf(f_stat, n_params - 1, n - n_params)
    return f_stat, p_value

# Gestational week models
print("Gestational Week Models:")
f_stat_gw_linear, p_val_gw_linear = f_test_significance(y_target, y_pred_linear_gw, 2)
print(f"Linear model: F-statistic = {f_stat_gw_linear:.4f}, p-value = {p_val_gw_linear:.4e}")

f_stat_gw_poly2, p_val_gw_poly2 = f_test_significance(y_target, y_pred_poly2_gw, 3)
print(f"Poly2 model: F-statistic = {f_stat_gw_poly2:.4f}, p-value = {p_val_gw_poly2:.4e}")

f_stat_gw_poly3, p_val_gw_poly3 = f_test_significance(y_target, y_pred_poly3_gw, 4)
print(f"Poly3 model: F-statistic = {f_stat_gw_poly3:.4f}, p-value = {p_val_gw_poly3:.4e}")

# BMI models
print("\nBMI Models:")
f_stat_bmi_linear, p_val_bmi_linear = f_test_significance(y_target, y_pred_linear_bmi, 2)
print(f"Linear model: F-statistic = {f_stat_bmi_linear:.4f}, p-value = {p_val_bmi_linear:.4e}")

f_stat_bmi_poly2, p_val_bmi_poly2 = f_test_significance(y_target, y_pred_poly2_bmi, 3)
print(f"Poly2 model: F-statistic = {f_stat_bmi_poly2:.4f}, p-value = {p_val_bmi_poly2:.4e}")

f_stat_bmi_poly3, p_val_bmi_poly3 = f_test_significance(y_target, y_pred_poly3_bmi, 4)
print(f"Poly3 model: F-statistic = {f_stat_bmi_poly3:.4f}, p-value = {p_val_bmi_poly3:.4e}")

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Y concentration vs Gestational Week
ax1 = axes[0, 0]
ax1.scatter(gestational_week, y_concentration, alpha=0.6, color='blue', s=50)

# Plot fitted lines
gw_range = np.linspace(gestational_week.min(), gestational_week.max(), 100)
X_gw_range = gw_range.reshape(-1, 1)

# Linear fit
y_linear_range = linear_model_gw.predict(X_gw_range)
ax1.plot(gw_range, y_linear_range, 'r-', label=f'Linear (R²={r2_linear_gw:.3f})', linewidth=2)

# Polynomial fit (best R²)
if r2_poly3_gw > r2_poly2_gw and r2_poly3_gw > r2_linear_gw:
    X_poly3_range = poly3_features_gw.transform(X_gw_range)
    y_poly3_range = poly3_model_gw.predict(X_poly3_range)
    ax1.plot(gw_range, y_poly3_range, 'g-', label=f'Poly3 (R²={r2_poly3_gw:.3f})', linewidth=2)
elif r2_poly2_gw > r2_linear_gw:
    X_poly2_range = poly2_features_gw.transform(X_gw_range)
    y_poly2_range = poly2_model_gw.predict(X_poly2_range)
    ax1.plot(gw_range, y_poly2_range, 'g-', label=f'Poly2 (R²={r2_poly2_gw:.3f})', linewidth=2)

ax1.set_xlabel('Gestational Week')
ax1.set_ylabel('Y Chromosome Concentration')
ax1.set_title('Y Chromosome Concentration vs Gestational Week')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Y concentration vs BMI
ax2 = axes[0, 1]
ax2.scatter(bmi, y_concentration, alpha=0.6, color='orange', s=50)

# Plot fitted lines
bmi_range = np.linspace(bmi.min(), bmi.max(), 100)
X_bmi_range = bmi_range.reshape(-1, 1)

# Linear fit
y_linear_range_bmi = linear_model_bmi.predict(X_bmi_range)
ax2.plot(bmi_range, y_linear_range_bmi, 'r-', label=f'Linear (R²={r2_linear_bmi:.3f})', linewidth=2)

# Polynomial fit (best R²)
if r2_poly3_bmi > r2_poly2_bmi and r2_poly3_bmi > r2_linear_bmi:
    X_poly3_range_bmi = poly3_features_bmi.transform(X_bmi_range)
    y_poly3_range_bmi = poly3_model_bmi.predict(X_poly3_range_bmi)
    ax2.plot(bmi_range, y_poly3_range_bmi, 'g-', label=f'Poly3 (R²={r2_poly3_bmi:.3f})', linewidth=2)
elif r2_poly2_bmi > r2_linear_bmi:
    X_poly2_range_bmi = poly2_features_bmi.transform(X_bmi_range)
    y_poly2_range_bmi = poly2_model_bmi.predict(X_poly2_range_bmi)
    ax2.plot(bmi_range, y_poly2_range_bmi, 'g-', label=f'Poly2 (R²={r2_poly2_bmi:.3f})', linewidth=2)

ax2.set_xlabel('BMI')
ax2.set_ylabel('Y Chromosome Concentration')
ax2.set_title('Y Chromosome Concentration vs BMI')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals for Gestational Week (best model)
ax3 = axes[1, 0]
if r2_poly3_gw > r2_poly2_gw and r2_poly3_gw > r2_linear_gw:
    residuals_gw = y_target - y_pred_poly3_gw
    fitted_values = y_pred_poly3_gw
    title_suffix = "Polynomial (degree 3)"
elif r2_poly2_gw > r2_linear_gw:
    residuals_gw = y_target - y_pred_poly2_gw
    fitted_values = y_pred_poly2_gw
    title_suffix = "Polynomial (degree 2)"
else:
    residuals_gw = y_target - y_pred_linear_gw
    fitted_values = y_pred_linear_gw
    title_suffix = "Linear"

ax3.scatter(fitted_values, residuals_gw, alpha=0.6)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Fitted Values')
ax3.set_ylabel('Residuals')
ax3.set_title(f'Residuals vs Fitted (GW, {title_suffix})')
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals for BMI (best model)
ax4 = axes[1, 1]
if r2_poly3_bmi > r2_poly2_bmi and r2_poly3_bmi > r2_linear_bmi:
    residuals_bmi = y_target - y_pred_poly3_bmi
    fitted_values_bmi = y_pred_poly3_bmi
    title_suffix_bmi = "Polynomial (degree 3)"
elif r2_poly2_bmi > r2_linear_bmi:
    residuals_bmi = y_target - y_pred_poly2_bmi
    fitted_values_bmi = y_pred_poly2_bmi
    title_suffix_bmi = "Polynomial (degree 2)"
else:
    residuals_bmi = y_target - y_pred_linear_bmi
    fitted_values_bmi = y_pred_linear_bmi
    title_suffix_bmi = "Linear"

ax4.scatter(fitted_values_bmi, residuals_bmi, alpha=0.6)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel('Fitted Values')
ax4.set_ylabel('Residuals')
ax4.set_title(f'Residuals vs Fitted (BMI, {title_suffix_bmi})')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('p1_y_conc_analysis.png', dpi=300, bbox_inches='tight')
print("Comprehensive analysis plot saved as 'p1_y_conc_analysis.png'")

# Additional individual plots with higher quality
plt.figure(figsize=(10, 6))
plt.scatter(gestational_week, y_concentration, alpha=0.7, color='blue', s=60)

# Plot the best fitting line
if r2_poly3_gw > r2_poly2_gw and r2_poly3_gw > r2_linear_gw:
    gw_smooth = np.linspace(gestational_week.min(), gestational_week.max(), 200)
    X_gw_smooth = gw_smooth.reshape(-1, 1)
    X_poly3_smooth = poly3_features_gw.transform(X_gw_smooth)
    y_poly3_smooth = poly3_model_gw.predict(X_poly3_smooth)
    plt.plot(gw_smooth, y_poly3_smooth, 'red', linewidth=3, 
             label=f'Cubic fit (R²={r2_poly3_gw:.4f})')
    best_model_gw = "Cubic"
    best_r2_gw = r2_poly3_gw
elif r2_poly2_gw > r2_linear_gw:
    gw_smooth = np.linspace(gestational_week.min(), gestational_week.max(), 200)
    X_gw_smooth = gw_smooth.reshape(-1, 1)
    X_poly2_smooth = poly2_features_gw.transform(X_gw_smooth)
    y_poly2_smooth = poly2_model_gw.predict(X_poly2_smooth)
    plt.plot(gw_smooth, y_poly2_smooth, 'red', linewidth=3, 
             label=f'Quadratic fit (R²={r2_poly2_gw:.4f})')
    best_model_gw = "Quadratic"
    best_r2_gw = r2_poly2_gw
else:
    gw_smooth = np.linspace(gestational_week.min(), gestational_week.max(), 200)
    X_gw_smooth = gw_smooth.reshape(-1, 1)
    y_linear_smooth = linear_model_gw.predict(X_gw_smooth)
    plt.plot(gw_smooth, y_linear_smooth, 'red', linewidth=3, 
             label=f'Linear fit (R²={r2_linear_gw:.4f})')
    best_model_gw = "Linear"
    best_r2_gw = r2_linear_gw

plt.xlabel('Gestational Week', fontsize=12)
plt.ylabel('Y Chromosome Concentration', fontsize=12)
plt.title('Fetal Y Chromosome Concentration vs Gestational Week', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('p1_y_conc_vs_gest_week.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(bmi, y_concentration, alpha=0.7, color='orange', s=60)

# Plot the best fitting line
if r2_poly3_bmi > r2_poly2_bmi and r2_poly3_bmi > r2_linear_bmi:
    bmi_smooth = np.linspace(bmi.min(), bmi.max(), 200)
    X_bmi_smooth = bmi_smooth.reshape(-1, 1)
    X_poly3_smooth_bmi = poly3_features_bmi.transform(X_bmi_smooth)
    y_poly3_smooth_bmi = poly3_model_bmi.predict(X_poly3_smooth_bmi)
    plt.plot(bmi_smooth, y_poly3_smooth_bmi, 'red', linewidth=3, 
             label=f'Cubic fit (R²={r2_poly3_bmi:.4f})')
    best_model_bmi = "Cubic"
    best_r2_bmi = r2_poly3_bmi
elif r2_poly2_bmi > r2_linear_bmi:
    bmi_smooth = np.linspace(bmi.min(), bmi.max(), 200)
    X_bmi_smooth = bmi_smooth.reshape(-1, 1)
    X_poly2_smooth_bmi = poly2_features_bmi.transform(X_bmi_smooth)
    y_poly2_smooth_bmi = poly2_model_bmi.predict(X_poly2_smooth_bmi)
    plt.plot(bmi_smooth, y_poly2_smooth_bmi, 'red', linewidth=3, 
             label=f'Quadratic fit (R²={r2_poly2_bmi:.4f})')
    best_model_bmi = "Quadratic"
    best_r2_bmi = r2_poly2_bmi
else:
    bmi_smooth = np.linspace(bmi.min(), bmi.max(), 200)
    X_bmi_smooth = bmi_smooth.reshape(-1, 1)
    y_linear_smooth_bmi = linear_model_bmi.predict(X_bmi_smooth)
    plt.plot(bmi_smooth, y_linear_smooth_bmi, 'red', linewidth=3, 
             label=f'Linear fit (R²={r2_linear_bmi:.4f})')
    best_model_bmi = "Linear"
    best_r2_bmi = r2_linear_bmi

plt.xlabel('BMI', fontsize=12)
plt.ylabel('Y Chromosome Concentration', fontsize=12)
plt.title('Fetal Y Chromosome Concentration vs Maternal BMI', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('p1_y_conc_vs_bmi.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"Best model for Gestational Week: {best_model_gw} (R² = {best_r2_gw:.4f})")
print(f"Best model for BMI: {best_model_bmi} (R² = {best_r2_bmi:.4f})")

print(f"\nCorrelation with Gestational Week: r = {corr_gw:.4f} (p = {p_value_gw:.4e})")
print(f"Correlation with BMI: r = {corr_bmi:.4f} (p = {p_value_bmi:.4e})")

significance_level = 0.05
print(f"\nSignificance test results (α = {significance_level}):")
print(f"Gestational Week correlation: {'Significant' if p_value_gw < significance_level else 'Not significant'}")
print(f"BMI correlation: {'Significant' if p_value_bmi < significance_level else 'Not significant'}")

# Save detailed results to file
with open('y_chromosome_analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write("Y Chromosome Concentration Analysis Results\n")
    f.write("="*50 + "\n\n")
    
    f.write("Dataset Information:\n")
    f.write(f"- Total samples analyzed: {len(df_clean)}\n")
    f.write(f"- Gestational week range: {gestational_week.min():.1f} - {gestational_week.max():.1f}\n")
    f.write(f"- BMI range: {bmi.min():.1f} - {bmi.max():.1f}\n")
    f.write(f"- Y concentration range: {y_concentration.min():.6f} - {y_concentration.max():.6f}\n\n")
    
    f.write("Correlation Analysis:\n")
    f.write(f"- Gestational Week vs Y concentration: r = {corr_gw:.4f}, p = {p_value_gw:.4e}\n")
    f.write(f"- BMI vs Y concentration: r = {corr_bmi:.4f}, p = {p_value_bmi:.4e}\n\n")
    
    f.write("Model Performance (R² values):\n")
    f.write("Gestational Week Models:\n")
    f.write(f"- Linear: {r2_linear_gw:.4f}\n")
    f.write(f"- Quadratic: {r2_poly2_gw:.4f}\n")
    f.write(f"- Cubic: {r2_poly3_gw:.4f}\n\n")
    
    f.write("BMI Models:\n")
    f.write(f"- Linear: {r2_linear_bmi:.4f}\n")
    f.write(f"- Quadratic: {r2_poly2_bmi:.4f}\n")
    f.write(f"- Cubic: {r2_poly3_bmi:.4f}\n\n")
    
    f.write("Best Fitting Models:\n")
    f.write(f"- Gestational Week: {best_model_gw} (R² = {best_r2_gw:.4f})\n")
    f.write(f"- BMI: {best_model_bmi} (R² = {best_r2_bmi:.4f})\n")

print("\nAnalysis complete! Results saved to 'y_chromosome_analysis_results.txt'")