# 导入所有必需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet # 导入多种线性回归模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # 导入强大的集成学习模型
from sklearn.svm import SVR # 导入支持向量回归
from sklearn.preprocessing import StandardScaler # 用于数据标准化
from sklearn.cluster import KMeans # 用于聚类
from sklearn.model_selection import cross_val_score # 用于交叉验证
from sklearn.metrics import r2_score # 用于计算R²分数，评估回归模型性能
from scipy.stats import pearsonr, spearmanr # 用于统计检验
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置字体以正确显示中英文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 数据加载和初步清洗 ---
print("="*80)
print("高级Y染色体浓度分析 - 这是一个独立的、更深入的分析脚本")
print("="*80)

# 从Excel文件中专门读取名为'男胎检测数据'的工作表
df_male = pd.read_excel('/home/tfisher/code/math/2025/c-problem/附件.xlsx', sheet_name='男胎检测数据')

def parse_gestational_week(gw_str):
    """一个健壮的函数，用于将'11w+6'这样的字符串解析为数值（例如11 + 6/7）。"""
    if pd.isna(gw_str) or gw_str == '': return np.nan
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

# 应用上述函数，创建数值型的孕周列
df_male['检测孕周_numeric'] = df_male['检测孕周'].apply(parse_gestational_week)

# 定义我们希望用于建模的特征列
feature_cols = ['检测孕周_numeric', '孕妇BMI', 'Y染色体浓度', 'GC含量', '年龄', '体重', '身高']
# 仅保留数据文件中实际存在的列
available_cols = [col for col in feature_cols if col in df_male.columns]
df_clean = df_male[available_cols].dropna() # 删除任何包含缺失值的行，确保数据质量

# --- 第一阶段：数据预处理和动态分组 ---
print("\n第一阶段：数据预处理和动态BMI分组")

def improved_outlier_detection(data, factor=3.5):
    """使用改良的Tukey方法来检测异常值，factor=3.5比传统的1.5更宽松，适合有偏分布。"""
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data >= lower_bound) & (data <= upper_bound)

# 对关键变量应用异常值检测
valid_mask = improved_outlier_detection(df_clean['孕妇BMI']) & \
             improved_outlier_detection(df_clean['检测孕周_numeric']) & \
             improved_outlier_detection(df_clean['Y染色体浓度'])
df_clean = df_clean[valid_mask]
print(f"移除异常值后剩余样本数: {len(df_clean)}")

# 将清理后的数据提取为Numpy数组，方便后续计算
gestational_week = df_clean['检测孕周_numeric'].values
bmi = df_clean['孕妇BMI'].values
y_concentration = df_clean['Y染色体浓度'].values
gc_content = df_clean.get('GC含量', 0.42 + 0.003 * gestational_week) # 如果GC含量不存在，则创建一个模拟的
age = df_clean.get('年龄', 28 + np.random.normal(0, 4, len(gestational_week))) # 同上，模拟年龄

# --- 第二阶段：特征工程与高级回归 ---
print("\n第二阶段：特征工程与高级回归模型构建")

# 目标：通过构建大量有意义的特征，来更准确地预测Y染色体浓度。
# 这是一个创造性的过程，目的是从原始数据中榨取更多信息。
features = {
    'GW': gestational_week,
    'BMI': bmi,
    'GC': gc_content,
    'Age': age,
    'GW*BMI': gestational_week * bmi, # 交互项：孕周和BMI的共同影响
    'GW*GC': gestational_week * gc_content,
    'BMI*GC': bmi * gc_content,
    'GW²': gestational_week**2, # 多项式项：捕捉非线性关系
    'BMI²': bmi**2,
    'LogBMI': np.log(bmi + 1), # 对数变换：处理偏态分布
    'SqrtGW': np.sqrt(gestational_week) # 平方根变换
}
X_enhanced = pd.DataFrame(features)
y_target = y_concentration

# 标准化特征：使得所有特征都在同一尺度上，这对很多模型（如SVR、正则化线性模型）至关重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enhanced)

print(f"构建了 {X_scaled.shape[1]} 个特征用于模型训练。")

# --- 第三阶段：多种模型训练、评估与集成 ---
print("\n第三阶段：模型训练、评估与选择")

# 定义一个包含多种强大回归模型的字典
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01, max_iter=2000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100)
}

best_model_name = ''
best_r2 = -1
best_model = None

# 循环训练和评估字典中的每个模型
for name, model in models.items():
    try:
        # 使用5折交叉验证来评估模型的泛化能力
        cv_scores = cross_val_score(model, X_scaled, y_target, cv=5, scoring='r2')
        cv_mean_r2 = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X_scaled, y_target)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y_target, y_pred)
        
        print(f"模型: {name:16s} | R² (训练集): {r2:.4f} | R² (交叉验证): {cv_mean_r2:.4f}")
        
        # 记录表现最好的模型
        if cv_mean_r2 > best_r2:
            best_r2 = cv_mean_r2
            best_model_name = name
            best_model = model
            
    except Exception as e:
        print(f"模型: {name:16s} | 训练失败: {e}")

print(f"\n表现最佳的模型是: {best_model_name}，交叉验证 R² = {best_r2:.4f}")

# --- 结果可视化与分析 ---
print("\n生成分析图表...")
best_model.fit(X_scaled, y_target)
best_predictions = best_model.predict(X_scaled)
final_r2 = r2_score(y_target, best_predictions)

plt.figure(figsize=(15, 8))
plt.suptitle(f'高级Y染色体浓度分析 (最佳模型: {best_model_name}, 最终 R² = {final_r2:.4f})', fontsize=16)

# 图1: 真实值 vs. 预测值
plt.subplot(2, 3, 1)
plt.scatter(y_target, best_predictions, alpha=0.6, edgecolors='k', s=50)
plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], 'r--', lw=2, label='理想情况')
plt.xlabel('真实的Y染色体浓度')
plt.ylabel('预测的Y染色体浓度')
plt.title('真实值 vs. 预测值')
plt.grid(True, alpha=0.5)
plt.legend()

# 图2: 残差图 (预测值 vs. 误差)
plt.subplot(2, 3, 2)
residuals = y_target - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.6, edgecolors='k', s=50)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差 (真实值 - 预测值)')
plt.title('残差图')
plt.grid(True, alpha=0.5)

# 图3: 特征重要性（如果模型支持）
plt.subplot(2, 3, 3)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_enhanced.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
    plt.title('Top 10 特征重要性')
else:
    plt.text(0.5, 0.5, '此模型不提供特征重要性', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('特征重要性')

# 图4: 孕周与浓度的关系
plt.subplot(2, 3, 4)
sns.regplot(x=gestational_week, y=y_target, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='趋势线')
plt.xlabel('孕周')
plt.ylabel('Y染色体浓度')
plt.title('孕周与Y染色体浓度的关系')
plt.grid(True, alpha=0.5)

# 图5: BMI与浓度的关系
plt.subplot(2, 3, 5)
sns.regplot(x=bmi, y=y_target, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='趋势线')
plt.xlabel('孕妇BMI')
plt.ylabel('Y染色体浓度')
plt.title('BMI与Y染色体浓度的关系')
plt.grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('new-plan/advanced_y_chromosome_analysis.png', dpi=300)
plt.show()

print("\n分析完成。")
