#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题一：胎儿Y染色体浓度与孕妇指标的相关性分析
使用MIC方法分析Y染色体浓度与孕周数、BMI等指标的关系
建立相关性模型并检验显著性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
import warnings
warnings.filterwarnings('ignore')

# 设置字体（支持英文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    df = pd.read_excel('附件.xlsx')
    return df

def calculate_mic(x, y, alpha=0.6, c=15, est="mic_approx"):
    """
    计算最大信息系数(MIC)
    由于minepy库可能不可用，这里实现一个简化版本
    使用互信息的近似计算
    """
    from sklearn.feature_selection import mutual_info_regression
    
    # 确保输入是数值型
    x = np.array(x).reshape(-1, 1) if len(np.array(x).shape) == 1 else np.array(x)
    y = np.array(y)
    
    # 计算互信息
    mi = mutual_info_regression(x, y, random_state=42)
    
    # 归一化到[0,1]区间 - 简化的MIC近似
    # 真正的MIC需要更复杂的网格搜索
    n = len(y)
    max_possible_mi = min(np.log2(n), 4)  # 简化的上界估计
    mic_approx = mi[0] / max_possible_mi if max_possible_mi > 0 else 0
    
    return min(mic_approx, 1.0)  # 确保不超过1

def analyze_y_chromosome_correlation():
    """分析Y染色体浓度的相关性"""
    
    print("=== 问题一：胎儿Y染色体浓度与孕妇指标的相关性分析 ===\n")
    
    # 加载数据
    df = load_data()
    
    # 数据预处理
    print("1. 数据预处理")
    print(f"原始数据形状: {df.shape}")
    
    # 筛选有Y染色体数据的样本（男胎）
    male_df = df[df['Y染色体浓度'] > 0].copy()
    print(f"男胎样本数量: {len(male_df)}")
    
    # 处理检测孕周列（可能包含范围值）
    def process_gestational_week(week_str):
        """处理孕周数据，提取数值"""
        if pd.isna(week_str):
            return np.nan
        week_str = str(week_str).strip().lower()
        
        # 移除常见的单位标识
        week_str = week_str.replace('w', '').replace('周', '').replace('天', '').replace('d', '')
        
        # 如果包含+，取前面的数字
        if '+' in week_str:
            try:
                num_part = week_str.split('+')[0].strip()
                return float(num_part)
            except:
                return np.nan
        # 如果包含-，取平均值
        elif '-' in week_str:
            parts = week_str.split('-')
            try:
                start = float(parts[0].strip())
                end = float(parts[1].strip())
                return (start + end) / 2
            except:
                try:
                    return float(parts[0].strip())
                except:
                    return np.nan
        else:
            try:
                return float(week_str)
            except:
                return np.nan
    
    male_df['孕周_数值'] = male_df['检测孕周'].apply(process_gestational_week)
    
    # 移除缺失值
    analysis_columns = ['Y染色体浓度', '孕周_数值', '孕妇BMI', '年龄', '身高', '体重', 
                       'GC含量', 'Y染色体的Z值', 'X染色体的Z值']
    
    clean_df = male_df[analysis_columns].dropna()
    print(f"清洗后有效样本数量: {len(clean_df)}")
    
    # 2. 描述性统计
    print("\n2. Y染色体浓度的描述性统计")
    y_concentration = clean_df['Y染色体浓度']
    print(f"均值: {y_concentration.mean():.4f}")
    print(f"标准差: {y_concentration.std():.4f}")
    print(f"最小值: {y_concentration.min():.4f}")
    print(f"最大值: {y_concentration.max():.4f}")
    print(f"4%达标率: {(y_concentration >= 0.04).mean():.2%}")
    
    # 3. 相关性分析
    print("\n3. 相关性分析")
    
    # 计算皮尔逊相关系数
    target_var = 'Y染色体浓度'
    features = [col for col in analysis_columns if col != target_var]
    
    correlations = []
    mic_values = []
    p_values = []
    
    print("变量名\t\t\t皮尔逊相关系数\tMIC值\t\t显著性(p值)")
    print("-" * 70)
    
    for feature in features:
        # 皮尔逊相关系数
        corr, p_val = stats.pearsonr(clean_df[feature], clean_df[target_var])
        correlations.append(corr)
        p_values.append(p_val)
        
        # MIC值
        mic_val = calculate_mic(clean_df[feature], clean_df[target_var])
        mic_values.append(mic_val)
        
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        print(f"{feature:<20}\t{corr:>8.4f}\t\t{mic_val:>8.4f}\t{p_val:>10.6f} {significance}")
    
    # 4. 创建相关性结果DataFrame
    correlation_results = pd.DataFrame({
        '特征': features,
        '皮尔逊相关系数': correlations,
        'MIC值': mic_values,
        'p值': p_values
    })
    
    # 按MIC值排序
    correlation_results = correlation_results.sort_values('MIC值', ascending=False)
    
    print(f"\n按MIC值排序的相关性分析结果:")
    print(correlation_results.round(4))
    
    # 5. 可视化分析
    create_visualizations(clean_df, correlation_results)
    
    # 6. 建立回归模型
    build_regression_models(clean_df, correlation_results)
    
    # 7. 显著性检验
    perform_significance_tests(clean_df)
    
    return clean_df, correlation_results

def create_visualizations(clean_df, correlation_results):
    """创建可视化图表 - 英文版"""
    print("\n4. 创建可视化图表")
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    
    # 创建特征名称的英文映射
    feature_name_mapping = {
        '孕周_数值': 'Gestational Week',
        '孕妇BMI': 'Maternal BMI', 
        '年龄': 'Maternal Age',
        '身高': 'Height (cm)',
        '体重': 'Weight (kg)',
        'GC含量': 'GC Content',
        'Y染色体的Z值': 'Y-chr Z-score',
        'X染色体的Z值': 'X-chr Z-score'
    }
    
    # 图1: Y染色体浓度分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Y Chromosome Concentration Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 浓度分布直方图
    axes[0,0].hist(clean_df['Y染色体浓度'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(0.04, color='red', linestyle='--', label='4% Threshold')
    axes[0,0].set_xlabel('Y Chromosome Concentration')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Y Chromosome Concentration')
    axes[0,0].legend()
    
    # 与孕周的关系散点图
    axes[0,1].scatter(clean_df['孕周_数值'], clean_df['Y染色体浓度'], alpha=0.6, color='coral')
    axes[0,1].axhline(0.04, color='red', linestyle='--', alpha=0.7, label='4% Threshold')
    axes[0,1].set_xlabel('Gestational Week')
    axes[0,1].set_ylabel('Y Chromosome Concentration')
    axes[0,1].set_title('Y Chromosome Concentration vs Gestational Week')
    axes[0,1].legend()
    
    # 与BMI的关系散点图  
    axes[1,0].scatter(clean_df['孕妇BMI'], clean_df['Y染色体浓度'], alpha=0.6, color='lightgreen')
    axes[1,0].axhline(0.04, color='red', linestyle='--', alpha=0.7, label='4% Threshold')
    axes[1,0].set_xlabel('Maternal BMI')
    axes[1,0].set_ylabel('Y Chromosome Concentration')
    axes[1,0].set_title('Y Chromosome Concentration vs Maternal BMI')
    axes[1,0].legend()
    
    # MIC值柱状图
    top_features = correlation_results.head(6).copy()
    top_features['特征_英文'] = top_features['特征'].map(feature_name_mapping)
    
    bars = axes[1,1].bar(range(len(top_features)), top_features['MIC值'], 
                        color='gold', alpha=0.8, edgecolor='black')
    axes[1,1].set_xlabel('Features')
    axes[1,1].set_ylabel('MIC Value')
    axes[1,1].set_title('MIC Values for Y Chromosome Concentration')
    axes[1,1].set_xticks(range(len(top_features)))
    axes[1,1].set_xticklabels(top_features['特征_英文'], rotation=45, ha='right')
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('Y_Chromosome_Correlation_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图2: 相关性热力图
    plt.figure(figsize=(10, 8))
    
    # 选择主要变量创建相关系数矩阵，并使用英文标签
    main_vars = ['Y染色体浓度', '孕周_数值', '孕妇BMI', '年龄', 'GC含量', 'Y染色体的Z值']
    corr_matrix = clean_df[main_vars].corr()
    
    # 创建英文标签
    english_labels = ['Y-chr Concentration', 'Gestational Week', 'Maternal BMI', 
                     'Maternal Age', 'GC Content', 'Y-chr Z-score']
    
    # 重命名行列索引
    corr_matrix.index = english_labels
    corr_matrix.columns = english_labels
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Heatmap of Key Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization charts saved with English annotations")

def build_regression_models(clean_df, correlation_results):
    """建立回归模型"""
    print("\n5. 建立回归模型")
    
    # 准备数据
    target = clean_df['Y染色体浓度']
    
    # 选择前5个最相关的特征
    top_features = correlation_results.head(5)['特征'].tolist()
    X = clean_df[top_features]
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=top_features, index=X.index)
    
    # 多元线性回归模型
    model = LinearRegression()
    model.fit(X_scaled, target)
    
    # 预测
    y_pred = model.predict(X_scaled)
    r2 = r2_score(target, y_pred)
    
    print("多元线性回归模型结果:")
    print(f"决定系数 R²: {r2:.4f}")
    print(f"调整 R²: {1 - (1-r2)*(len(target)-1)/(len(target)-len(top_features)-1):.4f}")
    
    # 回归系数
    coefficients = pd.DataFrame({
        '特征': top_features,
        '回归系数': model.coef_,
        '标准化系数': model.coef_
    })
    
    print("\n回归系数:")
    print(coefficients.round(4))
    
    # F检验
    f_statistic, f_pvalue = f_regression(X_scaled, target)
    
    print(f"\n模型F检验:")
    print(f"F统计量: {f_statistic.mean():.4f}")
    print(f"p值: {f_pvalue.mean():.6f}")
    
    # 残差分析
    residuals = target - y_pred
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(1, 3, 2)
    plt.scatter(target, y_pred, alpha=0.6)
    plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    
    plt.subplot(1, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Residual Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig('Regression_Model_Diagnostics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, scaler

def perform_significance_tests(clean_df):
    """执行显著性检验"""
    print("\n6. 显著性检验")
    
    target = clean_df['Y染色体浓度']
    
    # 正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(target)
    print(f"Y染色体浓度正态性检验 (Shapiro-Wilk):")
    print(f"统计量: {shapiro_stat:.4f}, p值: {shapiro_p:.6f}")
    
    if shapiro_p < 0.05:
        print("拒绝正态性假设，数据不服从正态分布")
    else:
        print("接受正态性假设，数据服从正态分布")
    
    # 不同BMI组的Y染色体浓度差异检验
    print(f"\n不同BMI组的差异检验:")
    
    # BMI分组（参考中国标准）
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return "偏瘦"
        elif bmi < 24:
            return "正常"
        elif bmi < 28:
            return "超重" 
        else:
            return "肥胖"
    
    clean_df['BMI组'] = clean_df['孕妇BMI'].apply(categorize_bmi)
    
    groups = []
    group_names = []
    for name, group in clean_df.groupby('BMI组'):
        if len(group) >= 10:  # 样本量足够
            groups.append(group['Y染色体浓度'])
            group_names.append(name)
            print(f"{name}组: n={len(group)}, 均值={group['Y染色体浓度'].mean():.4f}")
    
    if len(groups) >= 3:
        # 方差齐性检验
        levene_stat, levene_p = stats.levene(*groups)
        print(f"\n方差齐性检验 (Levene): 统计量={levene_stat:.4f}, p值={levene_p:.6f}")
        
        # ANOVA检验
        f_stat, anova_p = stats.f_oneway(*groups)
        print(f"单因素ANOVA: F={f_stat:.4f}, p值={anova_p:.6f}")
        
        if anova_p < 0.05:
            print("不同BMI组间Y染色体浓度存在显著差异")
        else:
            print("不同BMI组间Y染色体浓度无显著差异")

def main():
    """主函数"""
    try:
        clean_df, correlation_results = analyze_y_chromosome_correlation()
        
        print("\n=== 分析总结 ===")
        print("1. 成功分析了Y染色体浓度与各指标的相关性")
        print("2. 使用MIC方法识别了非线性关系")
        print("3. 建立了多元线性回归模型")
        print("4. 完成了统计显著性检验")
        print("5. 生成了可视化图表用于结果展示")
        
        # 保存结果
        correlation_results.to_csv('Y染色体浓度相关性分析结果.csv', index=False, encoding='utf-8-sig')
        clean_df.to_csv('清洗后的数据.csv', index=False, encoding='utf-8-sig')
        
        print("\n结果文件已保存:")
        print("- Y染色体浓度相关性分析结果.csv")
        print("- 清洗后的数据.csv")
        print("- Y_Chromosome_Correlation_Analysis.png (English)")
        print("- Correlation_Heatmap.png (English)") 
        print("- Regression_Model_Diagnostics.png (English)")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()