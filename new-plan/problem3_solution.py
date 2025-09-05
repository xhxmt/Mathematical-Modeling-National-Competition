# 导入所有必需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold  # 用于分层K折交叉验证
from sklearn.metrics import roc_auc_score  # 用于计算AUC分数，评估模型性能
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
import warnings

# 忽略所有警告信息
warnings.filterwarnings('ignore')

# 设置matplotlib的全局字体和图形大小
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class Problem3_StratifiedValidation:
    """
    这是一个为解决问题三（分层验证与动态风险量化）而设计的类。
    这个问题的核心是使用一种叫做“Cox比例风险模型”的统计方法来评估风险，
    并动态地更新模型参数，使其更加精确。
    """

    def __init__(self, data_path="/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv"):
        """
        类的构造函数。
        """
        self.df = pd.read_csv(data_path)
        self.setup_data()

    def setup_data(self):
        """
        为问题三准备和预处理数据。
        这里的重点是创建“时间依赖型协变量”，即随时间变化的特征。
        """
        # --- 提取孕周数值 ---
        def extract_week(week_str):
            if pd.isna(week_str): return np.nan
            try:
                return float(str(week_str).split('w')[0])
            except:
                return np.nan
        self.df['gestational_week'] = self.df['检测孕周'].apply(extract_week)

        # --- 数据筛选 ---
        # 筛选出拥有完整数据的男性胎儿样本
        self.male_data = self.df[
            (self.df['Y_concentration'] > 0) &
            (self.df['gestational_week'].notna()) &
            (self.df['gestational_week'] >= 12) &
            (self.df['gestational_week'] <= 28) &
            (self.df['孕妇BMI'].notna()) &
            (self.df['GC含量'].notna())
        ].copy()

        # --- 创建时间依赖型特征 ---
        # 1. 标准化BMI：将BMI值进行标准化（减去平均值后除以标准差），使其不受量纲影响
        self.male_data['BMI_t'] = (self.male_data['孕妇BMI'] - self.male_data['孕妇BMI'].mean()) / self.male_data['孕妇BMI'].std()

        # 2. 计算GC含量变化（Delta GC）：对于每个孕妇，计算本次检测与上次检测的GC含量差异
        self.male_data = self.male_data.sort_values(['孕妇代码', 'gestational_week']) # 必须先按孕妇和时间排序
        self.male_data['Delta_GC'] = self.male_data.groupby('孕妇代码')['GC含量'].diff().fillna(0) # .diff()计算差值

        # 3. 标准化Z值：对Y染色体的Z值进行标准化
        self.male_data['Z_t'] = (self.male_data['Y染色体的Z值'] - self.male_data['Y染色体的Z值'].mean()) / self.male_data['Y染色体的Z值'].std()

        # --- 定义“事件” ---
        # 在生存分析中，“事件”通常指我们关心的坏结果。在这里，我们定义“事件”为“Y染色体浓度未能达到4%的阈值”。
        self.male_data['Y_above_threshold'] = self.male_data['Y_concentration'] >= 0.04
        self.male_data['event'] = (~self.male_data['Y_above_threshold']).astype(int)  # event=1表示发生了坏事（未达标）

        print(f"用于问题三分析的样本数: {len(self.male_data)}")
        print(f"事件发生率 (Y染色体浓度未达标的比例): {self.male_data['event'].mean()*100:.2f}%")

    def cox_proportional_hazards_approximation(self):
        """
        使用逻辑回归来近似Cox比例风险模型。
        Cox模型本身比较复杂，但在某些情况下，逻辑回归可以作为一个很好的近似，
        用来估计各个风险因素（如BMI, Delta_GC, Z_t）的影响程度（即β系数）。
        """
        # 准备特征X和目标y
        X = self.male_data[['BMI_t', 'Delta_GC', 'Z_t']].fillna(0)
        y = self.male_data['event']

        # 创建并训练逻辑回归模型
        cox_model = LogisticRegression(random_state=42, max_iter=1000)
        cox_model.fit(X, y)

        # 从训练好的模型中提取β系数
        beta_1, beta_2, beta_3 = cox_model.coef_[0]
        intercept = cox_model.intercept_[0] # 截距项可以看作是基础风险

        # 计算每个样本的风险分数
        risk_scores = cox_model.predict_proba(X)[:, 1] # 取值为1（事件发生）的概率

        print("近似Cox模型得到的系数:")
        print(f"β₁ (BMI的影响): {beta_1:.4f}")
        print(f"β₂ (ΔGC的影响): {beta_2:.4f}")
        print(f"β₃ (Z值的影响): {beta_3:.4f}")

        return cox_model, (beta_1, beta_2, beta_3, intercept), risk_scores

    def optimal_detection_time_decision(self, cox_model, beta_params):
        """
        构建并求解一个双目标优化问题，来决定最优的检测时间。
        目标：max [S(T) - λ·Risk(T)]
        S(T)是“生存概率”，即到T时刻仍未发生坏事的概率，我们希望它大。
        Risk(T)是与时间相关的风险惩罚，我们希望它小。
        λ是一个权重，用来平衡这两者。
        """
        beta_1, beta_2, beta_3, intercept = beta_params
        lambda_penalty = 0.3  # 风险惩罚的权重

        optimal_times = []

        # 为数据集中的每个样本单独计算最优检测时间
        for _, row in self.male_data.iterrows():
            bmi_t, delta_gc, z_t = row['BMI_t'], row['Delta_GC'], row['Z_t']

            def dual_objective(T):
                # 生存概率的近似计算
                survival_prob = 1 / (1 + np.exp(intercept + beta_1*bmi_t + beta_2*delta_gc + beta_3*z_t))

                # 风险惩罚函数，对过早、过晚的检测进行惩罚
                risk_penalty = 0
                if T < 12: risk_penalty = 0.5
                elif 12 <= T <= 22: risk_penalty = 0.1 * (T - 16)**2
                elif T > 22: risk_penalty = 1.0

                return survival_prob - lambda_penalty * risk_penalty

            # 在[12, 28]周的范围内寻找使目标函数最大的T
            T_range = np.linspace(12, 28, 100)
            objectives = [dual_objective(T) for T in T_range]
            optimal_T = T_range[np.argmax(objectives)]
            optimal_times.append(optimal_T)

        self.male_data['optimal_detection_time'] = optimal_times
        return optimal_times

    def stratified_cross_validation(self, cox_model):
        """
        进行分层交叉验证，并计算在problem3.md中定义的“组间一致性指数 (ICI)”。
        分层交叉验证可以更鲁棒地评估模型性能。ICI则用来衡量模型在不同数据子集上表现得是否一致。
        """
        X = self.male_data[['BMI_t', 'Delta_GC', 'Z_t']].fillna(0)
        y = self.male_data['event']

        # 使用StratifiedKFold进行5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        group_aucs, group_early_scores = [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # 计算AUC分数
            group_aucs.append(roc_auc_score(y_test, y_pred_proba))

            # 计算EARLY分数（一个自定义的，衡量提前检测有效性的指标）
            test_weeks = self.male_data.iloc[test_idx]['gestational_week']
            optimal_weeks = self.male_data.iloc[test_idx]['optimal_detection_time']
            early_score = np.mean((optimal_weeks < test_weeks) * np.exp(-0.1 * (test_weeks - optimal_weeks)))
            group_early_scores.append(early_score)

        # 计算汇总指标
        pooled_auc = np.mean(group_aucs)
        pooled_early = np.mean(group_early_scores)

        # 计算ICI指数，ICI越接近1，说明模型表现越稳定一致
        ici = 1 - np.mean((np.abs(np.array(group_aucs) - pooled_auc) + np.abs(np.array(group_early_scores) - pooled_early)) / (pooled_auc + pooled_early))

        print("\n分层交叉验证结果:")
        print(f"汇总AUC: {pooled_auc:.4f}, 汇总EARLY分数: {pooled_early:.4f}, 组间一致性指数(ICI): {ici:.4f}")

        return {'group_aucs': group_aucs, 'group_early_scores': group_early_scores, 'pooled_auc': pooled_auc, 'pooled_early': pooled_early, 'ici': ici}

    def bayesian_parameter_update(self, beta_params, n_bootstrap=50):
        """
        使用Bootstrap方法（一种重复抽样技术）来模拟贝叶斯参数更新。
        这可以帮助我们了解参数的不确定性，并得到一个更稳健的参数估计。
        """
        beta_1, beta_2, beta_3, intercept = beta_params
        alpha = 0.9  # 记忆因子，表示我们多大程度上相信旧的参数

        bootstrap_betas = []
        X = self.male_data[['BMI_t', 'Delta_GC', 'Z_t']].fillna(0)
        y = self.male_data['event']

        # 进行50次Bootstrap抽样
        for i in range(n_bootstrap):
            boot_indices = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X.iloc[boot_indices], y.iloc[boot_indices]

            try:
                model = LogisticRegression(random_state=i, max_iter=1000).fit(X_boot, y_boot)
                bootstrap_betas.append(list(model.coef_[0]) + [model.intercept_[0]])
            except:
                continue

        bootstrap_betas = np.array(bootstrap_betas)

        # 贝叶斯更新规则：新参数 = α * 旧参数 + (1-α) * 从数据中学到的新信息
        updated_params = alpha * np.array(beta_params) + (1 - alpha) * np.mean(bootstrap_betas, axis=0)
        param_uncertainty = np.std(bootstrap_betas, axis=0) # 用标准差来衡量参数的不确定性

        print("\n贝叶斯参数更新结果:")
        print(f"更新后 β₁: {updated_params[0]:.4f} ± {param_uncertainty[0]:.4f}")
        print(f"更新后 β₂: {updated_params[1]:.4f} ± {param_uncertainty[1]:.4f}")
        print(f"更新后 β₃: {updated_params[2]:.4f} ± {param_uncertainty[2]:.4f}")

        return updated_params, param_uncertainty

    def visualize_results(self, cv_results, beta_params, updated_params):
        """为问题三创建一套完整的可视化图表。"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('问题三：分层验证与动态风险量化', fontsize=16)

        # 图1: 交叉验证中每次的AUC分数
        ax1 = axes[0, 0]
        ax1.bar(range(len(cv_results['group_aucs'])), cv_results['group_aucs'], color='lightblue', edgecolor='black')
        ax1.axhline(y=cv_results['pooled_auc'], color='red', linestyle='--', label=f"平均AUC: {cv_results['pooled_auc']:.3f}")
        ax1.set_title('交叉验证AUC分数')
        ax1.legend()

        # 图2: 交叉验证中每次的EARLY分数
        ax2 = axes[0, 1]
        ax2.bar(range(len(cv_results['group_early_scores'])), cv_results['group_early_scores'], color='lightgreen', edgecolor='black')
        ax2.axhline(y=cv_results['pooled_early'], color='red', linestyle='--', label=f"平均EARLY: {cv_results['pooled_early']:.3f}")
        ax2.set_title('提前检测有效性(EARLY)分数')
        ax2.legend()

        # 图3: 参数更新前后的对比
        ax3 = axes[1, 0]
        param_names = ['β₁ (BMI)', 'β₂ (ΔGC)', 'β₃ (Z)', '截距']
        x = np.arange(len(param_names))
        ax3.bar(x - 0.2, beta_params, 0.4, label='原始参数', color='orange')
        ax3.bar(x + 0.2, updated_params, 0.4, label='贝叶斯更新后', color='purple')
        ax3.set_title('参数更新前后对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(param_names, rotation=45)
        ax3.legend()

        # 图4: 计算出的最优检测时间的分布
        ax4 = axes[1, 1]
        ax4.hist(self.male_data['optimal_detection_time'], bins=20, color='skyblue', edgecolor='black')
        ax4.axvline(x=self.male_data['optimal_detection_time'].mean(), color='red', linestyle='--', label=f"平均值: {self.male_data['optimal_detection_time'].mean():.1f} 周")
        ax4.set_title('最优检测时间的分布')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('new-plan/problem3_analysis.png', dpi=300)
        plt.show()

    def run_analysis(self):
        """运行问题三的完整分析流程。"""
        print("=== 问题三：分层验证与动态风险量化分析开始 ===")

        cox_model, beta_params, risk_scores = self.cox_proportional_hazards_approximation()
        optimal_times = self.optimal_detection_time_decision(cox_model, beta_params)
        print(f"\n最优检测时间统计: 平均={np.mean(optimal_times):.2f}周, 标准差={np.std(optimal_times):.2f}周")

        cv_results = self.stratified_cross_validation(cox_model)
        updated_params, param_uncertainty = self.bayesian_parameter_update(beta_params)

        self.visualize_results(cv_results, beta_params, updated_params)

        # 保存结果到CSV
        results_df = pd.DataFrame({
            'Parameter': ['β₁ (BMI)', 'β₂ (ΔGC)', 'β₃ (Z)', 'Intercept'],
            'Original_Value': beta_params,
            'Updated_Value': updated_params[:4],
            'Uncertainty': param_uncertainty[:4]
        })
        results_df.to_csv('new-plan/problem3_results.csv', index=False)
        print("\n分析结果已保存到 problem3_results.csv")

if __name__ == "__main__":
    problem3 = Problem3_StratifiedValidation()
    problem3.run_analysis()
