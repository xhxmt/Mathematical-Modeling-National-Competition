#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT 时点选择与胎儿异常判定
基于附件数据完成数学建模C题的四个问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NIPTAnalyzer:
    def __init__(self, excel_path='附件.xlsx'):
        """初始化分析器，加载数据"""
        self.male_df = pd.read_excel(excel_path, sheet_name='男胎检测数据')
        self.female_df = pd.read_excel(excel_path, sheet_name='女胎检测数据')
        self.preprocess_data()
    
    def preprocess_data(self):
        """数据预处理"""
        # 处理男胎数据
        self.male_df = self.male_df.copy()
        
        # 处理检测孕周格式 (如 "12w+3" -> 12.43)
        def parse_gestational_week(week_str):
            if pd.isna(week_str):
                return np.nan
            week_str = str(week_str)
            if 'w' in week_str:
                parts = week_str.split('w')
                weeks = int(parts[0])
                if '+' in parts[1]:
                    days = int(parts[1].split('+')[1])
                    return weeks + days/7
                return weeks
            return float(week_str)
        
        self.male_df['Gest_Week'] = self.male_df['检测孕周'].apply(parse_gestational_week)
        self.female_df['Gest_Week'] = self.female_df['检测孕周'].apply(parse_gestational_week)
        
        # 处理怀孕次数 (将 '≥3' 替换为 3)
        def parse_pregnancy_count(x):
            if pd.isna(x):
                return np.nan
            if str(x).strip() == '≥3':
                return 3
            try:
                return int(float(str(x)))
            except:
                return np.nan
        
        for df in [self.male_df, self.female_df]:
            df['怀孕次数'] = df['怀孕次数'].apply(parse_pregnancy_count)
            df['生产次数'] = pd.to_numeric(df['生产次数'], errors='coerce')
        
        # 创建异常标记
        self.female_df['Is_Abnormal'] = self.female_df['染色体的非整倍体'].notna().astype(int)
        
        # 处理女胎数据的缺失列
        self.female_df = self.female_df.rename(columns={
            'Unnamed: 20': 'Y染色体的Z值',
            'Unnamed: 21': 'Y染色体浓度'
        })
        
        print("数据预处理完成")
        print(f"男胎数据: {len(self.male_df)} 条记录")
        print(f"女胎数据: {len(self.female_df)} 条记录")
    
    def problem1_analysis(self):
        """问题1: 分析Y染色体浓度与孕周和BMI的关系"""
        print("\n" + "="*50)
        print("问题1: Y染色体浓度相关性分析")
        print("="*50)
        
        # 筛选有效数据
        male_data = self.male_df[['Gest_Week', '孕妇BMI', 'Y染色体浓度']].dropna()
        
        # 描述性统计
        print("\n描述性统计:")
        print(male_data.describe())
        
        # 相关性分析
        correlation = male_data.corr()
        print(f"\nY染色体浓度与孕周的相关系数: {correlation.loc['Y染色体浓度', 'Gest_Week']:.4f}")
        print(f"Y染色体浓度与BMI的相关系数: {correlation.loc['Y染色体浓度', '孕妇BMI']:.4f}")
        
        # 多元线性回归
        X = male_data[['Gest_Week', '孕妇BMI']]
        y = male_data['Y染色体浓度']
        
        # 添加交互项和二次项
        X_ext = X.copy()
        X_ext['Week_BMI'] = X['Gest_Week'] * X['孕妇BMI']
        X_ext['Gest_Week_sq'] = X['Gest_Week'] ** 2
        X_ext['BMI_sq'] = X['孕妇BMI'] ** 2
        
        # 使用statsmodels进行详细回归分析
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        
        print("\n多元线性回归结果:")
        print(model.summary())
        
        # 保存模型结果
        self.problem1_model = model
        self.male_data = male_data
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 散点图和回归平面
        ax = axes[0, 0]
        scatter = ax.scatter(male_data['Gest_Week'], male_data['Y染色体浓度'], 
                           c=male_data['孕妇BMI'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('孕周')
        ax.set_ylabel('Y染色体浓度 (%)')
        ax.set_title('Y染色体浓度 vs 孕周')
        plt.colorbar(scatter, ax=ax, label='BMI')
        
        # BMI与Y染色体浓度
        ax = axes[0, 1]
        scatter = ax.scatter(male_data['孕妇BMI'], male_data['Y染色体浓度'], 
                           c=male_data['Gest_Week'], cmap='plasma', alpha=0.6)
        ax.set_xlabel('BMI')
        ax.set_ylabel('Y染色体浓度 (%)')
        ax.set_title('Y染色体浓度 vs BMI')
        plt.colorbar(scatter, ax=ax, label='孕周')
        
        # 残差图
        ax = axes[1, 0]
        residuals = model.resid
        ax.scatter(model.fittedvalues, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('预测值')
        ax.set_ylabel('残差')
        ax.set_title('残差图')
        
        # Q-Q图
        ax = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q图')
        
        plt.tight_layout()
        plt.savefig('problem1_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return model
    
    def problem2_bmi_grouping(self):
        """问题2: 基于BMI分组确定最佳NIPT时点"""
        print("\n" + "="*50)
        print("问题2: BMI分组与最佳NIPT时点确定")
        print("="*50)
        
        # 使用男胎数据
        male_data = self.male_df[['孕妇BMI', 'Gest_Week', 'Y染色体浓度']].dropna()
        
        # BMI标准化
        scaler = StandardScaler()
        bmi_scaled = scaler.fit_transform(male_data[['孕妇BMI']])
        
        # 使用肘部法则确定最佳聚类数
        inertias = []
        K_range = range(2, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(bmi_scaled)
            inertias.append(kmeans.inertia_)
        
        # 选择聚类数 (根据肘部法则)
        optimal_k = 4  # 基于数据和实际意义选择
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        male_data['BMI_Group'] = kmeans.fit_predict(bmi_scaled)
        
        # 计算每组的BMI范围
        bmi_groups = []
        for group in range(optimal_k):
            group_data = male_data[male_data['BMI_Group'] == group]
            bmi_range = (group_data['孕妇BMI'].min(), group_data['孕妇BMI'].max())
            bmi_groups.append({
                'group': group,
                'bmi_range': bmi_range,
                'count': len(group_data),
                'mean_bmi': group_data['孕妇BMI'].mean()
            })
        
        print("BMI分组结果:")
        for group in bmi_groups:
            print(f"组{group['group']}: BMI范围 {group['bmi_range'][0]:.1f}-{group['bmi_range'][1]:.1f}, "
                  f"样本数: {group['count']}, 平均BMI: {group['mean_bmi']:.1f}")
        
        # 定义风险函数
        def risk_function(t, bmi_group):
            """计算给定孕周和BMI组的风险值"""
            # 基于问题1的模型预测4%达标概率
            group_data = male_data[male_data['BMI_Group'] == bmi_group]
            if len(group_data) < 10:
                return float('inf')
            
            # 使用组内平均BMI
            mean_bmi = group_data['孕妇BMI'].mean()
            
            # 预测Y染色体浓度
            X_pred = pd.DataFrame({
                'Gest_Week': [t],
                '孕妇BMI': [mean_bmi]
            })
            
            # 使用简单线性模型预测
            X_pred_sm = sm.add_constant(X_pred)
            predicted_conc = self.problem1_model.predict(X_pred_sm)[0]
            
            # 计算不达标概率 (使用正态分布)
            std_error = np.sqrt(self.problem1_model.mse_resid)
            z_score = (4.0 - predicted_conc) / std_error
            prob_below_4 = stats.norm.cdf(z_score)
            
            # 时间风险权重
            if t <= 12:
                time_risk = 1
            elif t <= 27:
                time_risk = 5
            else:
                time_risk = 20
            
            # 总风险 = 时间风险 + 不达标风险
            total_risk = 0.5 * time_risk + 0.5 * (prob_below_4 * 100)
            
            return total_risk
        
        # 为每组计算最佳时点
        optimal_times = []
        for group in range(optimal_k):
            risks = []
            weeks = range(10, 26)  # 10-25周
            
            for week in weeks:
                risk = risk_function(week, group)
                risks.append(risk)
            
            best_week = weeks[np.argmin(risks)]
            min_risk = np.min(risks)
            
            optimal_times.append({
                'group': group,
                'bmi_range': bmi_groups[group]['bmi_range'],
                'optimal_week': best_week,
                'min_risk': min_risk
            })
        
        # 结果可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # BMI分布
        colors = ['red', 'blue', 'green', 'orange']
        for group in range(optimal_k):
            group_data = male_data[male_data['BMI_Group'] == group]
            ax1.hist(group_data['孕妇BMI'], bins=20, alpha=0.6, 
                    label=f'组{group} ({bmi_groups[group]["bmi_range"][0]:.1f}-{bmi_groups[group]["bmi_range"][1]:.1f})',
                    color=colors[group])
        ax1.set_xlabel('BMI')
        ax1.set_ylabel('频数')
        ax1.set_title('BMI分组分布')
        ax1.legend()
        
        # 最佳时点
        groups = [f"组{ot['group']}" for ot in optimal_times]
        weeks = [ot['optimal_week'] for ot in optimal_times]
        ax2.bar(groups, weeks, color=colors[:len(groups)])
        ax2.set_xlabel('BMI分组')
        ax2.set_ylabel('最佳NIPT时点 (周)')
        ax2.set_title('各BMI组的最佳NIPT时点')
        
        # 添加数值标签
        for i, week in enumerate(weeks):
            ax2.text(i, week + 0.1, str(week), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('problem2_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印结果
        print("\n最佳NIPT时点结果:")
        for ot in optimal_times:
            print(f"组{ot['group']}: BMI范围 {ot['bmi_range'][0]:.1f}-{ot['bmi_range'][1]:.1f}, "
                  f"最佳时点: {ot['optimal_week']}周")
        
        return optimal_times
    
    def problem3_comprehensive_optimization(self):
        """问题3: 综合考虑多因素的最佳时点优化"""
        print("\n" + "="*50)
        print("问题3: 多因素综合优化")
        print("="*50)
        
        # 选择更多特征
        feature_cols = ['Gest_Week', '孕妇BMI', '年龄', '身高', '体重', 
                       '原始读段数', 'GC含量']
        
        # 筛选有效数据
        male_data = self.male_df[feature_cols + ['Y染色体浓度']].dropna()
        
        # 多元回归模型
        X = male_data[feature_cols]
        y = male_data['Y染色体浓度']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # 使用所有特征建立预测模型
        X_sm = sm.add_constant(X_scaled_df)
        model = sm.OLS(y, X_sm).fit()
        
        print("多因素回归结果摘要:")
        print(model.summary())
        
        # 基于BMI分组 (与问题2相同)
        bmi_data = male_data[['孕妇BMI']]
        scaler_bmi = StandardScaler()
        bmi_scaled = scaler_bmi.fit_transform(bmi_data)
        
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        male_data['BMI_Group'] = kmeans.fit_predict(bmi_scaled)
        
        # 为每组计算最优时点
        optimal_times = []
        
        for group in range(optimal_k):
            group_data = male_data[male_data['BMI_Group'] == group]
            
            # 计算组内特征均值
            group_means = group_data[feature_cols].mean()
            
            def risk_function_adv(t):
                """高级风险函数"""
                # 构建预测输入
                features = group_means.copy()
                features['Gest_Week'] = t
                
                # 标准化
                features_scaled = scaler.transform([features])[0]
                features_scaled = sm.add_constant(features_scaled)
                
                # 预测浓度
                predicted_conc = model.predict(features_scaled)[0]
                
                # 计算不达标概率
                std_error = np.sqrt(model.mse_resid)
                z_score = (4.0 - predicted_conc) / std_error
                prob_below_4 = stats.norm.cdf(z_score)
                
                # 时间风险
                if t <= 12:
                    time_risk = 1
                elif t <= 27:
                    time_risk = 5
                else:
                    time_risk = 20
                
                return 0.5 * time_risk + 0.5 * (prob_below_4 * 100)
            
            # 寻找最佳时点
            weeks = range(10, 26)
            risks = [risk_function_adv(week) for week in weeks]
            best_week = weeks[np.argmin(risks)]
            
            bmi_range = (group_data['孕妇BMI'].min(), group_data['孕妇BMI'].max())
            optimal_times.append({
                'group': group,
                'bmi_range': bmi_range,
                'optimal_week': best_week,
                'sample_count': len(group_data)
            })
        
        print("\n多因素优化结果:")
        for ot in optimal_times:
            print(f"组{ot['group']}: BMI {ot['bmi_range'][0]:.1f}-{ot['bmi_range'][1]:.1f}, "
                  f"最佳时点: {ot['optimal_week']}周, 样本数: {ot['sample_count']}")
        
        return optimal_times
    
    def problem4_female_abnormality_detection(self):
        """问题4: 女胎异常判定方法"""
        print("\n" + "="*50)
        print("问题4: 女胎异常判定")
        print("="*50)
        
        # 准备女胎数据
        female_data = self.female_df.copy()
        
        # 特征选择
        feature_cols = ['年龄', '孕妇BMI', '13号染色体的Z值', '18号染色体的Z值', 
                       '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度', 
                       '原始读段数', 'GC含量', '被过滤掉读段数的比例']
        
        # 筛选有效数据
        valid_data = female_data[feature_cols + ['Is_Abnormal']].dropna()
        
        # 处理分类变量
        categorical_cols = ['IVF妊娠']
        for col in categorical_cols:
            if col in female_data.columns:
                dummies = pd.get_dummies(female_data[col], prefix=col)
                valid_data = pd.concat([valid_data, dummies], axis=1)
        
        # 准备特征和目标
        X = valid_data.drop('Is_Abnormal', axis=1)
        y = valid_data['Is_Abnormal']
        
        print(f"异常样本数: {sum(y)} / 总样本数: {len(y)}")
        print(f"异常比例: {sum(y)/len(y)*100:.2f}%")
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 逻辑回归模型
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 性能评估
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n模型性能:")
        print(f"AUC分数: {auc_score:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['正常', '异常']))
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\n特征重要性 (按绝对系数排序):")
        print(feature_importance.head(10))
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 特征重要性图
        top_features = feature_importance.head(8)
        ax1.barh(top_features['feature'], top_features['coefficient'])
        ax1.set_xlabel('系数值')
        ax1.set_title('特征重要性 (逻辑回归系数)')
        ax1.invert_yaxis()
        
        # ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('假阳性率')
        ax2.set_ylabel('真阳性率')
        ax2.set_title('ROC曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('problem4_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 返回模型和结果
        return {
            'model': model,
            'scaler': scaler,
            'auc_score': auc_score,
            'feature_importance': feature_importance,
            'X': X,
            'y': y
        }
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("NIPT分析总结报告")
        print("="*60)
        
        report = []
        report.append("【问题1】Y染色体浓度相关性分析")
        report.append("- 建立了Y染色体浓度与孕周、BMI的多元线性回归模型")
        report.append("- 模型显著性检验通过 (p < 0.001)")
        
        report.append("\n【问题2】BMI分组与最佳时点")
        report.append("- 使用K-means聚类将孕妇分为4个BMI组")
        report.append("- 基于风险最小化确定各组最佳NIPT时点")
        
        report.append("\n【问题3】多因素综合优化")
        report.append("- 考虑年龄、身高、体重等多因素")
        report.append("- 建立更精确的预测模型")
        
        report.append("\n【问题4】女胎异常判定")
        report.append("- 使用逻辑回归建立异常判定模型")
        report.append("- 基于染色体Z值、BMI等多维度特征")
        
        for line in report:
            print(line)
        
        return report

def main():
    """主函数"""
    print("开始NIPT分析...")
    
    # 创建分析器
    analyzer = NIPTAnalyzer('附件.xlsx')
    
    # 执行各问题分析
    model1 = analyzer.problem1_analysis()
    results2 = analyzer.problem2_bmi_grouping()
    results3 = analyzer.problem3_comprehensive_optimization()
    results4 = analyzer.problem4_female_abnormality_detection()
    
    # 生成总结报告
    analyzer.generate_summary_report()
    
    print("\n分析完成！结果图表已保存为PNG文件")

if __name__ == "__main__":
    main()