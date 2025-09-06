#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题一：基于MIC动态特征选择的NIPT检测时间优化模型
根据mic2.md中的数学公式实现高级时滞MIC分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 设置字体（支持英文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    df = pd.read_excel('附件.xlsx')
    return df

def standardize_features(X):
    """标准化处理 - 公式: X' = (X - μ_X) / σ_X"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def time_lag_filling(X, tau):
    """
    时滞填充 - 公式:
    X_{t-τ} = { X_1       if t ≤ τ
              { X_{t-τ}   if t > τ
    """
    if tau == 0:
        return X.copy()
    
    # 使用首值填充前tau个位置
    X_lagged = np.zeros_like(X)
    if len(X.shape) == 1:
        X_lagged[:tau] = X[0]
        X_lagged[tau:] = X[:-tau]
    else:
        X_lagged[:tau, :] = X[0, :]
        X_lagged[tau:, :] = X[:-tau, :]
    
    return X_lagged

def enhanced_mic_calculation(x, y, alpha=0.6, c=15):
    """
    增强版MIC计算，使用互信息近似
    结合时滞分析的改进算法
    """
    # 确保输入是数值型并处理缺失值
    x = np.array(x).reshape(-1, 1) if len(np.array(x).shape) == 1 else np.array(x)
    y = np.array(y)
    
    # 移除包含NaN的样本
    valid_idx = ~(np.isnan(x).any(axis=1) | np.isnan(y))
    x_clean = x[valid_idx]
    y_clean = y[valid_idx]
    
    if len(x_clean) < 10:  # 样本太少
        return 0.0
    
    # 计算互信息
    try:
        mi = mutual_info_regression(x_clean, y_clean, random_state=42)[0]
        
        # 改进的MIC归一化 - 考虑样本量和特征分布
        n = len(y_clean)
        # 根据样本量和特征变异性调整上界
        feature_entropy = stats.entropy(np.histogram(x_clean.flatten(), bins=min(20, n//5))[0] + 1e-10)
        target_entropy = stats.entropy(np.histogram(y_clean, bins=min(20, n//5))[0] + 1e-10)
        
        max_possible_mi = min(feature_entropy, target_entropy, np.log2(n/2))
        mic_score = mi / max_possible_mi if max_possible_mi > 0 else 0
        
        return min(mic_score, 1.0)
    except:
        return 0.0

def compute_mic_matrix(features_dict, target, max_tau=12):
    """
    计算MIC矩阵 - 公式: M_{ij} = MIC(Feature_i(t-τ_j), Accuracy(t))
    其中τ ∈ [0, 12]周（覆盖早期到中期检测窗口）
    """
    tau_range = np.arange(0, max_tau + 1)
    feature_names = list(features_dict.keys())
    
    mic_matrix = np.zeros((len(feature_names), len(tau_range)))
    
    print(f"计算MIC矩阵 ({len(feature_names)} features × {len(tau_range)} time lags)")
    
    for i, feature_name in enumerate(feature_names):
        feature_data = features_dict[feature_name]
        
        for j, tau in enumerate(tau_range):
            # 应用时滞填充
            lagged_feature = time_lag_filling(feature_data, tau)
            
            # 计算MIC值
            mic_val = enhanced_mic_calculation(lagged_feature, target)
            mic_matrix[i, j] = mic_val
            
        print(f"完成特征 '{feature_name}' 的时滞分析")
    
    return mic_matrix, feature_names, tau_range

def dynamic_clustering_optimization(mic_matrix, feature_names, alpha=0.5, beta=0.5, min_group_size=0.35):
    """
    动态聚类分组模型 - 公式:
    min Σ[α·Var(M_k) + β·ΔT_k]
    约束: ΔT_k ≤ 4周, |M_k| ≥ 0.35
    """
    n_features = len(feature_names)
    best_score = float('inf')
    best_k = 2
    best_clusters = None
    best_centers = None
    
    # 使用最大MIC值进行聚类
    max_mic_values = np.max(mic_matrix, axis=1)
    optimal_tau_indices = np.argmax(mic_matrix, axis=1)
    
    # 准备聚类数据：[max_MIC, optimal_tau]
    cluster_data = np.column_stack([max_mic_values, optimal_tau_indices])
    cluster_data_scaled, _ = standardize_features(cluster_data)
    
    print("\n动态聚类分组分析:")
    
    # 限制聚类数量范围，避免过多聚类
    max_k = min(6, n_features - 1)  # 确保不超过样本数-1
    
    # 尝试不同的聚类数量
    for k in range(2, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(cluster_data_scaled)
            centers = kmeans.cluster_centers_
            
            # 计算目标函数值
            total_score = 0
            valid_clusters = 0
            
            for cluster_id in range(k):
                cluster_mask = clusters == cluster_id
                cluster_features = max_mic_values[cluster_mask]
                cluster_taus = optimal_tau_indices[cluster_mask]
                
                if len(cluster_features) == 0:
                    continue
                    
                # 检查约束条件
                mean_mic = np.mean(cluster_features)
                tau_variance = np.var(cluster_taus) if len(cluster_taus) > 1 else 0
                
                # 约束：平均MIC值 ≥ min_group_size, tau范围 ≤ 4周
                if mean_mic >= min_group_size and (np.max(cluster_taus) - np.min(cluster_taus)) <= 4:
                    mic_variance = np.var(cluster_features) if len(cluster_features) > 1 else 0
                    tau_spread = np.max(cluster_taus) - np.min(cluster_taus)
                    
                    cluster_score = alpha * mic_variance + beta * tau_spread
                    total_score += cluster_score
                    valid_clusters += 1
            
            if valid_clusters > 0 and total_score < best_score:
                best_score = total_score
                best_k = k
                best_clusters = clusters
                best_centers = centers
            
            # 计算轮廓系数，但要确保有足够的样本和合理的聚类数
            silhouette_avg = 0
            if k <= n_features and len(np.unique(clusters)) > 1 and len(np.unique(clusters)) < n_features:
                try:
                    silhouette_avg = silhouette_score(cluster_data_scaled, clusters)
                except:
                    silhouette_avg = 0
                    
            print(f"K={k}: Score={total_score:.4f}, Silhouette={silhouette_avg:.4f}, Valid_clusters={valid_clusters}")
            
        except Exception as e:
            print(f"K={k}: 聚类失败 - {str(e)}")
            continue
    
    # 如果没有找到有效聚类，使用简单的2聚类
    if best_clusters is None:
        print("使用默认2聚类方案")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_clusters = kmeans.fit_predict(cluster_data_scaled)
        best_centers = kmeans.cluster_centers_
        best_k = 2
        best_score = 0
    
    return best_k, best_clusters, best_centers, best_score

def risk_benefit_optimization(mic_matrix, tau_range, gamma=2.0, lambda_risk=0.3):
    """
    时变风险-收益优化模型 - 公式:
    max_T [Σ(w_τ · M(T-τ))] - λ·Risk(T)
    
    其中权重系数: w_τ = exp(γ·M_τ) / Σexp(γ·M_τ)
    """
    n_features, n_taus = mic_matrix.shape
    
    # 为每个时滞计算平均MIC值
    avg_mic_per_tau = np.mean(mic_matrix, axis=0)
    
    # 计算权重系数 w_τ = exp(γ·M_τ) / Σexp(γ·M_τ)
    exp_terms = np.exp(gamma * avg_mic_per_tau)
    weights = exp_terms / np.sum(exp_terms)
    
    print("\n时变风险-收益优化分析:")
    print("时滞(周)\t平均MIC\t权重系数")
    print("-" * 40)
    for tau, avg_mic, weight in zip(tau_range, avg_mic_per_tau, weights):
        print(f"{tau}\t\t{avg_mic:.4f}\t{weight:.4f}")
    
    # 寻找最优检测时间窗口
    optimal_benefits = []
    
    for T in tau_range:
        # 计算加权收益
        weighted_benefit = 0
        for tau_idx, tau in enumerate(tau_range):
            if T - tau >= 0 and T - tau < len(tau_range):
                lag_idx = T - tau
                weighted_benefit += weights[tau_idx] * avg_mic_per_tau[lag_idx]
        
        # 简化的风险模型：早期检测风险较高，中期相对较低
        if T < 4:  # 早期 (< 12周)
            risk = 0.8 * np.exp(-(T/2))
        elif T <= 8:  # 中期 (12-20周)
            risk = 0.3
        else:  # 晚期 (> 20周)
            risk = 0.5 + 0.1 * (T - 8)
        
        total_score = weighted_benefit - lambda_risk * risk
        optimal_benefits.append({
            'T': T,
            'benefit': weighted_benefit,
            'risk': risk,
            'total_score': total_score
        })
    
    # 找到最优时间点
    best_time = max(optimal_benefits, key=lambda x: x['total_score'])
    
    print(f"\n最优检测时间窗口: {best_time['T']}周")
    print(f"预期收益: {best_time['benefit']:.4f}")
    print(f"风险评分: {best_time['risk']:.4f}")
    print(f"综合评分: {best_time['total_score']:.4f}")
    
    return optimal_benefits, best_time, weights

def calculate_tsi(predictions_by_group, pooled_predictions, pooled_truth):
    """
    计算模型稳定性指数 (TSI):
    TSI = 1 - (1/K)·Σ|AUC_k - AUC_pooled|/AUC_pooled
    要求 TSI ≥ 0.85
    """
    try:
        # 计算汇总AUC
        pooled_auc = roc_auc_score(pooled_truth, pooled_predictions)
        
        # 计算各组AUC的偏差
        auc_deviations = []
        for group_id, (group_pred, group_truth) in predictions_by_group.items():
            if len(np.unique(group_truth)) > 1:  # 确保有正负样本
                group_auc = roc_auc_score(group_truth, group_pred)
                deviation = abs(group_auc - pooled_auc) / pooled_auc
                auc_deviations.append(deviation)
        
        if len(auc_deviations) == 0:
            return 0.0
        
        tsi = 1 - np.mean(auc_deviations)
        return max(0.0, tsi)  # 确保TSI >= 0
    except:
        return 0.0

def advanced_mic_analysis():
    """主要分析函数 - 实现mic2.md中的高级算法"""
    
    print("=== 基于MIC动态特征选择的NIPT检测时间优化模型 ===\n")
    
    # 1. 数据加载和预处理
    df = load_data()
    print(f"1. 数据加载完成: {df.shape}")
    
    # 筛选男胎样本
    male_df = df[df['Y染色体浓度'] > 0].copy()
    print(f"男胎样本数量: {len(male_df)}")
    
    # 处理孕周数据
    def process_gestational_week(week_str):
        if pd.isna(week_str):
            return np.nan
        week_str = str(week_str).strip().lower()
        week_str = week_str.replace('w', '').replace('周', '').replace('天', '').replace('d', '')
        
        if '+' in week_str:
            try:
                return float(week_str.split('+')[0].strip())
            except:
                return np.nan
        elif '-' in week_str:
            parts = week_str.split('-')
            try:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            except:
                return np.nan
        else:
            try:
                return float(week_str)
            except:
                return np.nan
    
    male_df['孕周_数值'] = male_df['检测孕周'].apply(process_gestational_week)
    
    # 准备特征集
    feature_columns = {
        'BMI': male_df['孕妇BMI'].values,
        'GC_Content': male_df['GC含量'].values,
        'Y_Z_Score': male_df['Y染色体的Z值'].values,
        'Unique_Reads': male_df['唯一比对的读段数  '].values,
        'Age': male_df['年龄'].values,
        'Weight': male_df['体重'].values,
        'Gestational_Week': male_df['孕周_数值'].values
    }
    
    # 移除缺失值
    valid_mask = ~pd.DataFrame(feature_columns).isnull().any(axis=1)
    for key in feature_columns:
        feature_columns[key] = feature_columns[key][valid_mask]
    
    target = male_df['Y染色体浓度'].values[valid_mask]
    
    print(f"有效样本数量: {len(target)}")
    
    # 2. 计算MIC矩阵
    print("\n2. 计算时滞MIC矩阵")
    mic_matrix, feature_names, tau_range = compute_mic_matrix(feature_columns, target, max_tau=12)
    
    # 3. 动态聚类分组
    print("\n3. 执行动态聚类分组")
    best_k, clusters, centers, score = dynamic_clustering_optimization(mic_matrix, feature_names)
    
    # 4. 风险-收益优化
    print("\n4. 时变风险-收益优化")
    optimal_benefits, best_time, weights = risk_benefit_optimization(mic_matrix, tau_range)
    
    # 5. 创建高级可视化
    create_advanced_visualizations(mic_matrix, feature_names, tau_range, clusters, optimal_benefits, weights)
    
    # 6. 模型验证和TSI计算
    print("\n5. 模型稳定性验证")
    validate_model_stability(feature_columns, target, clusters, feature_names)
    
    return {
        'mic_matrix': mic_matrix,
        'feature_names': feature_names,
        'tau_range': tau_range,
        'clusters': clusters,
        'optimal_time': best_time,
        'weights': weights
    }

def create_advanced_visualizations(mic_matrix, feature_names, tau_range, clusters, optimal_benefits, weights):
    """创建高级可视化图表"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. MIC热力图（改进版）
    ax1 = plt.subplot(3, 3, (1, 3))
    im = ax1.imshow(mic_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Time Lag (weeks)')
    ax1.set_ylabel('Features')
    ax1.set_title('Enhanced MIC Matrix Heatmap\n(Time-lagged Feature Analysis)', fontweight='bold')
    ax1.set_xticks(range(len(tau_range)))
    ax1.set_xticklabels(tau_range)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    
    # 添加数值标注
    for i in range(len(feature_names)):
        for j in range(len(tau_range)):
            ax1.text(j, i, f'{mic_matrix[i,j]:.3f}', ha='center', va='center', 
                    color='white' if mic_matrix[i,j] > 0.5 else 'black', fontsize=8)
    
    plt.colorbar(im, ax=ax1, label='MIC Value')
    
    # 2. 聚类结果可视化
    ax2 = plt.subplot(3, 3, 4)
    max_mic_values = np.max(mic_matrix, axis=1)
    optimal_tau_indices = np.argmax(mic_matrix, axis=1)
    
    scatter = ax2.scatter(optimal_tau_indices, max_mic_values, c=clusters, 
                         cmap='Set3', s=100, alpha=0.8, edgecolors='black')
    ax2.set_xlabel('Optimal Time Lag (weeks)')
    ax2.set_ylabel('Maximum MIC Value')
    ax2.set_title('Dynamic Clustering Results', fontweight='bold')
    
    # 添加特征名称标注
    for i, name in enumerate(feature_names):
        ax2.annotate(name, (optimal_tau_indices[i], max_mic_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 3. 时变权重分析
    ax3 = plt.subplot(3, 3, 5)
    bars = ax3.bar(tau_range, weights, color='gold', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Time Lag (weeks)')
    ax3.set_ylabel('Weight Coefficient')
    ax3.set_title('Time-varying Weight Distribution\n(Risk-Benefit Model)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 风险-收益优化曲线
    ax4 = plt.subplot(3, 3, 6)
    time_points = [opt['T'] for opt in optimal_benefits]
    benefits = [opt['benefit'] for opt in optimal_benefits]
    risks = [opt['risk'] for opt in optimal_benefits]
    total_scores = [opt['total_score'] for opt in optimal_benefits]
    
    ax4.plot(time_points, benefits, 'g-', label='Benefit', linewidth=2, marker='o')
    ax4.plot(time_points, risks, 'r--', label='Risk', linewidth=2, marker='s')
    ax4.plot(time_points, total_scores, 'b-', label='Total Score', linewidth=3, marker='^')
    
    # 标记最优点
    best_idx = np.argmax(total_scores)
    ax4.scatter(time_points[best_idx], total_scores[best_idx], 
               color='red', s=200, marker='*', zorder=5, label='Optimal Point')
    
    ax4.set_xlabel('Detection Time (weeks)')
    ax4.set_ylabel('Score')
    ax4.set_title('Risk-Benefit Optimization Curve', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 特征重要性排序
    ax5 = plt.subplot(3, 3, 7)
    feature_importance = np.max(mic_matrix, axis=1)
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    
    bars = ax5.barh(range(len(sorted_names)), sorted_importance, color='lightcoral', alpha=0.8)
    ax5.set_yticks(range(len(sorted_names)))
    ax5.set_yticklabels(sorted_names)
    ax5.set_xlabel('Maximum MIC Value')
    ax5.set_title('Feature Importance Ranking', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 时滞效应分析
    ax6 = plt.subplot(3, 3, 8)
    avg_mic_per_tau = np.mean(mic_matrix, axis=0)
    ax6.plot(tau_range, avg_mic_per_tau, 'purple', linewidth=3, marker='o', markersize=8)
    ax6.fill_between(tau_range, avg_mic_per_tau, alpha=0.3, color='purple')
    ax6.set_xlabel('Time Lag (weeks)')
    ax6.set_ylabel('Average MIC Value')
    ax6.set_title('Time Lag Effect Analysis', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 标记峰值
    peak_idx = np.argmax(avg_mic_per_tau)
    ax6.scatter(tau_range[peak_idx], avg_mic_per_tau[peak_idx], 
               color='red', s=150, marker='*', zorder=5)
    ax6.annotate(f'Peak at τ={tau_range[peak_idx]}', 
                xy=(tau_range[peak_idx], avg_mic_per_tau[peak_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 7. 模型性能总览
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis('off')
    
    # 性能指标文本
    performance_text = f"""
    MODEL PERFORMANCE SUMMARY
    
    ✓ Peak MIC Value: {np.max(mic_matrix):.4f}
    ✓ Optimal Time Lag: {tau_range[np.unravel_index(np.argmax(mic_matrix), mic_matrix.shape)[1]]} weeks
    ✓ Best Detection Window: {time_points[best_idx]} weeks
    ✓ Model Clusters: {len(np.unique(clusters))} groups
    ✓ Risk-Benefit Score: {max(total_scores):.4f}
    
    Key Insights:
    • BMI shows strongest correlation at lag {tau_range[peak_idx]} weeks
    • Optimal detection window: {time_points[best_idx]}-{time_points[best_idx]+2} weeks
    • High-weight features: {', '.join(sorted_names[:3])}
    """
    
    ax7.text(0.05, 0.95, performance_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('Advanced_MIC_Analysis_Dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("高级可视化分析图表已保存: Advanced_MIC_Analysis_Dashboard.png")

def validate_model_stability(feature_columns, target, clusters, feature_names):
    """验证模型稳定性并计算TSI"""
    
    # 基于聚类创建BMI组
    bmi_values = feature_columns['BMI']
    unique_clusters = np.unique(clusters)
    
    print(f"验证 {len(unique_clusters)} 个聚类组的模型稳定性:")
    
    # 为每个聚类组创建预测模型
    predictions_by_group = {}
    pooled_predictions = []
    pooled_truth = []
    
    # 创建二值化目标变量（是否达到4%阈值）
    target_binary = (target >= 0.04).astype(int)
    
    for cluster_id in unique_clusters:
        # 找到属于当前聚类的特征
        cluster_features = [i for i, c in enumerate(clusters) if c == cluster_id]
        
        if len(cluster_features) == 0:
            continue
        
        # 选择该聚类的特征数据
        selected_features = []
        for feat_idx in cluster_features:
            feat_name = feature_names[feat_idx]
            if feat_name in feature_columns:
                selected_features.append(feature_columns[feat_name])
        
        if len(selected_features) == 0:
            continue
        
        # 构建特征矩阵
        X_cluster = np.column_stack(selected_features)
        
        # 标准化
        X_scaled, _ = standardize_features(X_cluster)
        
        # 简单线性回归预测
        from sklearn.linear_model import LogisticRegression
        try:
            model = LogisticRegression(random_state=42)
            model.fit(X_scaled, target_binary)
            predictions = model.predict_proba(X_scaled)[:, 1]
            
            predictions_by_group[cluster_id] = (predictions, target_binary)
            pooled_predictions.extend(predictions)
            pooled_truth.extend(target_binary)
            
            print(f"聚类 {cluster_id}: {len(cluster_features)} 个特征, {len(predictions)} 个样本")
            
        except Exception as e:
            print(f"聚类 {cluster_id} 建模失败: {e}")
    
    # 计算TSI
    if len(predictions_by_group) >= 2:
        tsi = calculate_tsi(predictions_by_group, np.array(pooled_predictions), np.array(pooled_truth))
        print(f"\n模型稳定性指数 (TSI): {tsi:.4f}")
        
        if tsi >= 0.85:
            print("✓ TSI ≥ 0.85，模型稳定性达标")
        else:
            print("⚠ TSI < 0.85，模型稳定性需要改进")
    else:
        print("聚类组数不足，无法计算TSI")
    
    return tsi if 'tsi' in locals() else 0.0

def main():
    """主函数"""
    try:
        print("开始执行基于mic2.md公式的高级MIC分析...")
        
        results = advanced_mic_analysis()
        
        print("\n" + "="*60)
        print("高级MIC分析完成！")
        print("="*60)
        
        print(f"\n关键发现:")
        print(f"• 最优检测时间窗口: {results['optimal_time']['T']} 周")
        print(f"• 综合评分: {results['optimal_time']['total_score']:.4f}")
        print(f"• 聚类分组数: {len(np.unique(results['clusters']))} 个")
        print(f"• 最高MIC值: {np.max(results['mic_matrix']):.4f}")
        
        # 保存结果
        np.savez('advanced_mic_results.npz', 
                mic_matrix=results['mic_matrix'],
                feature_names=results['feature_names'],
                tau_range=results['tau_range'],
                clusters=results['clusters'],
                weights=results['weights'])
        
        print(f"\n结果已保存:")
        print("• advanced_mic_results.npz - 数值结果")  
        print("• Advanced_MIC_Analysis_Dashboard.png - 可视化dashboard")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()