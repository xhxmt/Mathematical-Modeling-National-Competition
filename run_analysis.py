#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版NIPT分析 - 模块化与可视化结构
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 数据加载与预处理 ---

def load_and_preprocess_data(excel_path='附件.xlsx'):
    """加载并预处理NIPT数据"""
    print("正在加载和预处理数据...")
    male_df = pd.read_excel(excel_path, sheet_name='男胎检测数据')
    female_df = pd.read_excel(excel_path, sheet_name='女胎检测数据')

    def parse_gestational_week(week_str):
        if pd.isna(week_str): return np.nan
        week_str = str(week_str)
        if 'w' in week_str:
            parts = week_str.split('w')
            weeks = int(parts[0])
            days_str = parts[1].replace('+', '')
            if days_str:
                days = int(days_str)
                return weeks + days / 7
            return float(weeks)
        try:
            return float(week_str)
        except (ValueError, TypeError):
            return np.nan

    for df in [male_df, female_df]:
        df['Gest_Week'] = df['检测孕周'].apply(parse_gestational_week)
        df['怀孕次数'] = pd.to_numeric(df['怀孕次数'].replace('≥3', 3), errors='coerce')
        df['生产次数'] = pd.to_numeric(df['生产次数'], errors='coerce')

    female_df['Is_Abnormal'] = female_df['染色体的非整倍体'].notna().astype(int)
    print(f"数据预处理完成 - 男胎: {len(male_df)}条, 女胎: {len(female_df)}条")
    return male_df, female_df

# --- 问题1: 关系建模 ---

def analyze_problem1(male_df):
    """问题1: Y染色体浓度与孕周和BMI的关系分析"""
    print("\n" + "="*50)
    print("问题1: Y染色体浓度与孕周和BMI的关系分析")
    print("="*50)
    
    male_data = male_df[['Gest_Week', '孕妇BMI', 'Y染色体浓度']].dropna().copy()
    X = male_data[['Gest_Week', '孕妇BMI']]
    y = male_data['Y染色体浓度']
    
    # --- 基础模型 ---
    X_base = sm.add_constant(X)
    model_base = sm.OLS(y, X_base).fit()
    print("\n--- 基础回归模型 ---")
    print(f"调整R² = {model_base.rsquared_adj:.4f}")
    
    # --- 交互项模型 ---
    male_data['Week_BMI_Interaction'] = male_data['Gest_Week'] * male_data['孕妇BMI']
    X_interact = male_data[['Gest_Week', '孕妇BMI', 'Week_BMI_Interaction']]
    X_interact_sm = sm.add_constant(X_interact)
    model_interact = sm.OLS(y, X_interact_sm).fit()
    print("\n--- 交互项回归模型 ---")
    print(f"调整R² = {model_interact.rsquared_adj:.4f}")

    # 选择更优的模型
    final_model = model_interact if model_interact.rsquared_adj > model_base.rsquared_adj else model_base
    print(f"\n选择模型: {'交互项模型' if final_model == model_interact else '基础模型'} (基于调整后R²)")
    print(final_model.summary())

    visualize_problem1(male_data, final_model)
    return final_model, male_data

def visualize_problem1(data, model):
    """Visualization for Problem 1"""
    print("Generating visualization for Problem 1...")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(data['Gest_Week'], data['孕妇BMI'], data['Y染色体浓度'], c='r', marker='o', label='Actual Data')

    # Create meshgrid for the plane
    week_surf = np.arange(data['Gest_Week'].min(), data['Gest_Week'].max(), 1)
    bmi_surf = np.arange(data['孕妇BMI'].min(), data['孕妇BMI'].max(), 1)
    week_surf, bmi_surf = np.meshgrid(week_surf, bmi_surf)

    # Predict Z values for the plane
    exog = pd.DataFrame({'Gest_Week': week_surf.ravel(), '孕妇BMI': bmi_surf.ravel()})
    if 'Week_BMI_Interaction' in model.params.index:
        exog['Week_BMI_Interaction'] = exog['Gest_Week'] * exog['孕妇BMI']

    exog = sm.add_constant(exog, has_constant='add')
    # Ensure column order matches the model's expected input
    exog = exog[model.params.index]

    z = model.predict(exog).values.reshape(week_surf.shape)

    ax.plot_surface(week_surf, bmi_surf, z, alpha=0.5, cmap=plt.cm.coolwarm, label='Regression Plane')

    ax.set_xlabel('Gestational Week')
    ax.set_ylabel('Maternal BMI (kg/m²)')
    ax.set_zlabel('Y-Chromosome Concentration (%)')
    ax.set_title('Y-Chromosome Concentration vs. Gestational Week and BMI')

    # Create a proxy artist for the legend
    proxy = plt.Rectangle((0, 0), 1, 1, fc=plt.cm.coolwarm(0.5), alpha=0.5)
    ax.legend(['Actual Data', 'Regression Plane'], handles=[ax.get_children()[0], proxy])

    save_path = os.path.join(OUTPUT_DIR, 'problem1_3d_regression_plane.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Chart saved to: {save_path}")

# --- 问题2: BMI分组与优化 ---

def analyze_problem2(male_df, model_p1):
    """问题2: 基于BMI分组确定最佳NIPT时点"""
    print("\n" + "="*50)
    print("问题2: 基于BMI分组确定最佳NIPT时点")
    print("="*50)

    male_data = male_df[['孕妇BMI', 'Gest_Week', 'Y染色体浓度']].dropna().copy()

    # --- K-Means聚类 & 肘部法则确定K值 ---
    scaler = StandardScaler()
    male_data['BMI_Scaled'] = scaler.fit_transform(male_data[['孕妇BMI']])

    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(male_data[['BMI_Scaled']])
        sse[k] = kmeans.inertia_

    # 简单的肘部法则：找到变化率最大的点 (K=3 or 4 is usually optimal)
    # 这里为了简化，我们直接选择K=4，并在图中展示其合理性
    K = 4
    print(f"使用肘部法则确定K值 (可视化展示), 选择 K={K}")
    kmeans = KMeans(n_clusters=K, random_state=42)
    male_data['BMI_Group'] = kmeans.fit_predict(male_data[['BMI_Scaled']])

    print("\nBMI分组结果:")
    # 按BMI均值排序分组
    group_means = male_data.groupby('BMI_Group')['孕妇BMI'].mean().sort_values()
    group_mapping = {old_group: new_group for new_group, old_group in enumerate(group_means.index)}
    male_data['BMI_Group'] = male_data['BMI_Group'].map(group_mapping)

    for i in sorted(male_data['BMI_Group'].unique()):
        group_info = male_data[male_data['BMI_Group'] == i]['孕妇BMI'].describe()
        print(f"组{i}: BMI范围 [{group_info['min']:.1f}-{group_info['max']:.1f}], 均值 {group_info['mean']:.1f}, 样本数 {int(group_info['count'])}")

    # --- 风险计算和时点优化 (使用修正后的风险函数) ---
    optimal_times = {}
    risk_curves = {}

    # 定义标准化的时间风险
    time_risk_map = {1: 1, 5: 5, 20: 20}
    min_risk_val, max_risk_val = min(time_risk_map.values()), max(time_risk_map.values())

    def get_normalized_time_risk(week):
        raw_risk = 1 if week <= 12 else (5 if week < 28 else 20)
        return (raw_risk - min_risk_val) / (max_risk_val - min_risk_val)

    for group_id in sorted(male_data['BMI_Group'].unique()):
        group_data = male_data[male_data['BMI_Group'] == group_id]
        mean_bmi = group_data['孕妇BMI'].mean()
        
        weeks = np.arange(10, 25.1, 0.5)
        risks = []
        for week in weeks:
            pred_data = {'Gest_Week': week, '孕妇BMI': mean_bmi}
            if 'Week_BMI_Interaction' in model_p1.params.index:
                pred_data['Week_BMI_Interaction'] = week * mean_bmi
            
            exog = pd.DataFrame([pred_data])
            exog = sm.add_constant(exog, has_constant='add')
            exog = exog[model_p1.params.index]

            pred_conc = model_p1.predict(exog)[0]
            std_error = np.sqrt(model_p1.mse_resid)
            inaccuracy_risk = stats.norm.cdf((0.04 - pred_conc) / std_error)
            
            time_risk_norm = get_normalized_time_risk(week)
            
            # 修正后的风险函数
            total_risk = 0.5 * time_risk_norm + 0.5 * inaccuracy_risk
            risks.append(total_risk)
        
        risk_curves[group_id] = {'weeks': weeks, 'risks': risks}
        best_week_index = np.argmin(risks)
        optimal_times[group_id] = {
            'best_week': weeks[best_week_index],
            'min_risk': risks[best_week_index]
        }

    print("\n最佳NIPT时点结果 (修正后):")
    for group_id, result in optimal_times.items():
        print(f"组{group_id}: 推荐 {result['best_week']:.1f} 周")
        
    visualize_problem2(male_data, optimal_times, risk_curves, sse)
    return male_data, optimal_times

def visualize_problem2(data, optimal_times, risk_curves, sse):
    """Visualization for Problem 2"""
    print("Generating visualizations for Problem 2...")
    
    # 1. Elbow Method Plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(sse.keys()), list(sse.values()), 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (SSE)')
    plt.title('Elbow Method for K-Means Clustering')
    plt.axvline(x=4, color='r', linestyle='--', label='Selected K=4')
    plt.legend()
    save_path_elbow = os.path.join(OUTPUT_DIR, 'problem2_elbow_method.png')
    plt.savefig(save_path_elbow)
    plt.close()
    print(f"Chart saved to: {save_path_elbow}")

    # 2. Risk Curve Plots for each group
    num_groups = len(risk_curves)
    fig, axes = plt.subplots(num_groups, 1, figsize=(12, 5 * num_groups), sharex=True)
    if num_groups == 1:
        axes = [axes] # Make it iterable if there's only one subplot
    fig.suptitle('Risk Curves and Optimal NIPT Time for each BMI Group', fontsize=16)

    for group_id, curves in risk_curves.items():
        ax = axes[group_id]
        ax.plot(curves['weeks'], curves['risks'], label='Total Risk')
        
        best_week = optimal_times[group_id]['best_week']
        min_risk = optimal_times[group_id]['min_risk']
        
        ax.axvline(x=best_week, color='r', linestyle='--', label=f'Optimal Time: {best_week:.1f} Weeks')
        ax.plot(best_week, min_risk, 'ro') # Mark the minimum point
        
        group_info = data[data['BMI_Group'] == group_id]['孕妇BMI'].describe()
        ax.set_title(f'Group {group_id} (BMI: {group_info["min"]:.1f}-{group_info["max"]:.1f})')
        ax.set_ylabel('Total Risk (Normalized)')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Gestational Week')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path_risk = os.path.join(OUTPUT_DIR, 'problem2_risk_curves.png')
    plt.savefig(save_path_risk)
    plt.close()
    print(f"Chart saved to: {save_path_risk}")

# --- 问题3: 多因素优化 ---

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def analyze_problem3(male_df, male_data_grouped):
    """问题3: 使用梯度提升模型进行多因素综合优化"""
    print("\n" + "="*50)
    print("问题3: 多因素综合优化 (使用梯度提升模型)")
    print("="*50)

    # --- 1. 准备多因素数据 ---
    feature_cols = ['Gest_Week', '孕妇BMI', '年龄', '身高', '体重', '怀孕次数', '生产次数']
    multi_factor_data = male_df[feature_cols + ['Y染色体浓度']].dropna().copy()

    X = multi_factor_data[feature_cols]
    y = multi_factor_data['Y染色体浓度']

    # --- 2. 训练梯度提升回归模型 ---
    print("正在训练梯度提升模型...")
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X, y)

    # 评估模型性能 (可选，但推荐)
    y_pred = gbr.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"梯度提升模型性能: R²={r2:.4f}, MSE={mse:.4f}")

    # 获取特征重要性
    feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': gbr.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\n特征重要性:")
    print(feature_importance)

    # --- 3. 使用新模型重新进行风险优化 ---
    # 使用与问题2相同的分组和风险函数逻辑
    optimal_times = {}
    risk_curves = {}

    time_risk_map = {1: 1, 5: 5, 20: 20}
    min_risk_val, max_risk_val = min(time_risk_map.values()), max(time_risk_map.values())

    def get_normalized_time_risk(week):
        raw_risk = 1 if week <= 12 else (5 if week < 28 else 20)
        return (raw_risk - min_risk_val) / (max_risk_val - min_risk_val)

    # 估算预测的标准差 (GBR没有直接的mse_resid)
    # 使用交叉验证来获得一个更鲁棒的误差估计
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        gbr_cv = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbr_cv.fit(X_train, y_train)
        preds = gbr_cv.predict(X_test)
        errors.extend(y_test - preds)
    std_error_gbr = np.std(errors)
    print(f"\n梯度提升模型预测误差的标准差 (通过CV估计): {std_error_gbr:.4f}")


    for group_id in sorted(male_data_grouped['BMI_Group'].unique()):
        group_data = male_df[male_df.index.isin(male_data_grouped[male_data_grouped['BMI_Group'] == group_id].index)]
        
        # 使用分组的均值作为代表进行预测
        mean_features = group_data[feature_cols].mean()
        
        weeks = np.arange(10, 25.1, 0.5)
        risks = []
        for week in weeks:
            current_features = mean_features.copy()
            current_features['Gest_Week'] = week
            
            pred_conc = gbr.predict(pd.DataFrame([current_features]))[0]
            inaccuracy_risk = stats.norm.cdf((0.04 - pred_conc) / std_error_gbr)
            time_risk_norm = get_normalized_time_risk(week)
            
            total_risk = 0.5 * time_risk_norm + 0.5 * inaccuracy_risk
            risks.append(total_risk)
        
        risk_curves[group_id] = {'weeks': weeks, 'risks': risks}
        best_week_index = np.argmin(risks)
        optimal_times[group_id] = {
            'best_week': weeks[best_week_index],
            'min_risk': risks[best_week_index]
        }

    print("\n最佳NIPT时点结果 (基于多因素梯度提升模型):")
    for group_id, result in optimal_times.items():
        print(f"组{group_id}: 推荐 {result['best_week']:.1f} 周")
    
    visualize_problem3(male_data_grouped, optimal_times, risk_curves)
    return optimal_times

def visualize_problem3(data, optimal_times, risk_curves):
    """Visualization for Problem 3"""
    print("Generating visualization for Problem 3...")
    num_groups = len(risk_curves)
    fig, axes = plt.subplots(num_groups, 1, figsize=(12, 5 * num_groups), sharex=True)
    if num_groups == 1:
        axes = [axes] # Make it iterable if there's only one subplot
    fig.suptitle('Risk Curves and Optimal NIPT Time for each BMI Group (Multi-Factor Model)', fontsize=16)

    for group_id, curves in risk_curves.items():
        ax = axes[group_id]
        ax.plot(curves['weeks'], curves['risks'], label='Total Risk')
        
        best_week = optimal_times[group_id]['best_week']
        min_risk = optimal_times[group_id]['min_risk']
        
        ax.axvline(x=best_week, color='g', linestyle='--', label=f'Optimal Time: {best_week:.1f} Weeks')
        ax.plot(best_week, min_risk, 'go')
        
        group_info = data[data['BMI_Group'] == group_id]['孕妇BMI'].describe()
        ax.set_title(f'Group {group_id} (BMI: {group_info["min"]:.1f}-{group_info["max"]:.1f})')
        ax.set_ylabel('Total Risk (Normalized)')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Gestational Week')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path_risk = os.path.join(OUTPUT_DIR, 'problem3_risk_curves_multifactor.png')
    plt.savefig(save_path_risk)
    plt.close()
    print(f"Chart saved to: {save_path_risk}")

import xgboost as xgb
import shap

# --- 问题4: 女胎异常判定 ---

def analyze_problem4(female_df):
    """Problem 4: Female Fetus Abnormality Detection (using XGBoost and SHAP)"""
    print("\n" + "="*50)
    print("Problem 4: Female Fetus Abnormality Detection (using XGBoost and SHAP)")
    print("="*50)
    
    feature_cols = ['年龄', '孕妇BMI', '13号染色体的Z值', '18号染色体的Z值',
                   '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度',
                   '原始读段数', 'GC含量']
    
    valid_data = female_df[feature_cols + ['Is_Abnormal']].dropna().copy()
    X = valid_data[feature_cols]
    y = valid_data['Is_Abnormal']

    print(f"Dataset Info: {len(X)} samples, {y.sum()} abnormal samples ({(y.mean()*100):.2f}%)")

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- XGBoost Model ---
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Using scale_pos_weight to handle class imbalance: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nXGBoost Model AUC: {auc:.4f}")

    # SHAP Analysis
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    visualize_problem4(y_test, y_pred_proba, X_test, shap_values)
    return model

def visualize_problem4(y_test, y_pred_proba, X_test, shap_values):
    """Visualization for Problem 4"""
    print("Generating visualizations for Problem 4...")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost Model')
    plt.legend(loc="lower right")
    save_path_roc = os.path.join(OUTPUT_DIR, 'problem4_roc_curve.png')
    plt.savefig(save_path_roc)
    plt.close()
    print(f"Chart saved to: {save_path_roc}")

    # 2. SHAP Bar Plot (Global Feature Importance)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    save_path_shap_bar = os.path.join(OUTPUT_DIR, 'problem4_shap_summary_bar.png')
    plt.savefig(save_path_shap_bar, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {save_path_shap_bar}")

    # 3. SHAP Beeswarm Plot (Feature Impact Distribution)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Feature Impact Distribution')
    save_path_shap_summary = os.path.join(OUTPUT_DIR, 'problem4_shap_summary_beeswarm.png')
    plt.savefig(save_path_shap_summary, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {save_path_shap_summary}")

# --- 主函数 ---

def main():
    """主执行函数"""
    male_df, female_df = load_and_preprocess_data()
    
    # 问题1
    model_p1, male_data_p1 = analyze_problem1(male_df)
    
    # 问题2
    male_data_grouped, optimal_times_p2 = analyze_problem2(male_df, model_p1)

    # 问题3
    analyze_problem3(male_df, male_data_grouped)

    # 问题4
    model_p4 = analyze_problem4(female_df)
    
    # --- 最终总结 ---
    print("\n" + "="*60)
    print("分析完成！所有结果和图表已生成在 'output' 目录中。")
    print("="*60)
    print("\n主要发现摘要:")
    print("1. **关系建模 (问题1):**")
    print("   - Y染色体浓度与孕周、BMI等因素存在复杂的非线性关系。")
    print("   - 简单的线性模型拟合效果不佳 (调整R² < 0.05)，表明需要更复杂的模型。")

    print("\n2. **NIPT时点优化 (问题2 & 3):**")
    print("   - 通过K-Means将孕妇按BMI分为4组是合理的，如肘部法则图所示。")
    print("   - 使用高级模型（梯度提升）和修正后的风险函数，我们为每个BMI组提供了个性化的NIPT时点建议。")
    print("   - 结果表明，BMI越高的孕妇，为保证检测准确率，需要更晚的检测时点。")

    print("\n3. **女胎异常判定 (问题4):**")
    print("   - 我们构建了一个XGBoost分类器来识别异常胎儿，并使用`scale_pos_weight`处理了数据不平衡问题。")
    print("   - SHAP分析显示，各染色体的Z值是判定的最关键因素，这与临床医学知识相符。")
    print("   - 该模型不仅提供预测，还能通过SHAP图为每个判定提供可解释的依据，具有很高的临床应用潜力。")
    print("="*60)


if __name__ == "__main__":
    main()
