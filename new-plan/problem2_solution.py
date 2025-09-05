# 导入所有必需的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 用于更美观的统计图表
from scipy.optimize import minimize_scalar  # 从Scipy库导入一个用于标量函数优化的工具
from sklearn.linear_model import LogisticRegression # 从Scikit-learn库导入逻辑回归模型
import warnings  # 用于控制警告信息的显示

# 忽略所有警告信息，保持输出整洁
warnings.filterwarnings('ignore')

# 设置matplotlib的全局字体和图形大小
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class Problem2_TimeWindowOptimization:
    """
    这是一个专门为解决问题二（时间窗约束下的动态检测时间优化）而设计的类。
    它将所有相关的计算、分析和可视化都封装在了一起，使得整个流程更加清晰和模块化。
    这个模型的目标是，根据孕妇的BMI等指标，找到一个最佳的NIPT检测孕周。
    """

    def __init__(self, data_path="/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv"):
        """
        类的构造函数（初始化方法）。
        当创建这个类的一个实例时，这个方法会被自动调用。
        """
        # 从之前data_loader.py生成的已处理好的CSV文件中加载数据
        self.df = pd.read_csv(data_path)
        # 调用另一个方法，对数据进行针对问题二的特定设置
        self.setup_data()

    def setup_data(self):
        """
        为问题二准备和预处理数据。
        """
        # Y染色体浓度列在原始数据中已经是百分比格式，这里为了清晰，再赋值给一个新列
        self.df['Y_concentration_pct'] = self.df['Y_concentration']

        # 定义一个内部函数，用于从格式不统一的“检测孕周”列（如 "13w", "13w+6"）中提取周数
        def extract_week(week_str):
            if pd.isna(week_str):  # 如果值是空的，则返回空
                return np.nan
            try:
                # 尝试按'w'分割字符串，并取第一部分作为周数
                week_part = str(week_str).split('w')[0]
                return float(week_part)
            except:
                return np.nan # 如果转换失败，也返回空

        # 将上述函数应用到“检测孕周”列，生成一个只包含数值孕周的新列
        self.df['gestational_week'] = self.df['检测孕周'].apply(extract_week)

        # --- 数据筛选 ---
        # 根据问题要求，我们只分析男性胎儿，并且检测时间需要在临床可行范围内
        self.male_data = self.df[
            (self.df['Y_concentration_pct'] > 0) &  # Y染色体浓度大于0（代表男性胎儿）
            (self.df['gestational_week'].notna()) &  # 孕周数据不能为空
            (self.df['gestational_week'] >= 12) & # 孕周需大于等于12周
            (self.df['gestational_week'] <= 28)  # 孕周需小于等于28周
        ].copy() # 使用.copy()确保我们得到的是一个新的DataFrame，避免后续操作影响原始数据

        # 创建一个布尔列，标记Y染色体浓度是否达到4%的阈值
        self.male_data['Y_above_4pct'] = self.male_data['Y_concentration_pct'] >= 4.0

        # 计算GC含量的变化率（这里用标准化的GC含量作为代理）
        if len(self.male_data) > 0:
            self.male_data['GC_change'] = (self.male_data['GC含量'] - self.male_data['GC含量'].mean()) / self.male_data['GC含量'].std()

        print(f"筛选后用于问题二分析的有效男性胎儿样本数: {len(self.male_data)}")
        if len(self.male_data) > 0:
            print(f"Y染色体浓度范围: {self.male_data['Y_concentration_pct'].min():.6f} - {self.male_data['Y_concentration_pct'].max():.6f}")
            print(f"Y染色体浓度达到4%阈值的样本比例: {self.male_data['Y_above_4pct'].mean()*100:.2f}%")

    def comprehensive_benefit_function(self, T, BMI, GC_change, beta_params):
        """
        这个函数是问题二的核心，它实现了在problem2.md中定义的“综合效益函数”。
        Ψ(T) = [检测灵敏度] × [时间惩罚项] - [晚期惩罚项]
        函数的返回值越高，代表在孕周T进行检测的综合效益越大。
        """
        beta_0, beta_1, beta_2 = beta_params
        # 从问题描述中获取固定的权重和系数
        w_1, w_2, lambda_val = 2.0, 0.75, 0.85

        # 第一项：检测灵敏度。这是一个逻辑增长函数，通常认为随着孕周增加，检测会更准
        sensitivity = 1 / (1 + np.exp(-(beta_0 + beta_1 * BMI + beta_2 * GC_change)))

        # 第二项：时间惩罚。这个项表示检测时间越晚（但仍在22周内），效益越低
        if T <= 22:
            time_penalty = (1 - (T - 12) / (22 - 12)) ** w_1
        else:
            time_penalty = 0 # 超过22周，此项效益为0

        # 第三项：超晚期惩罚。如果检测时间超过28周，会有一个额外的惩罚
        late_penalty = 0
        if T > 28:
            late_penalty = lambda_val * (T / 40) ** w_2

        # 最终效益 = 灵敏度 * 时间惩罚 - 超晚期惩罚
        return sensitivity * time_penalty - late_penalty

    def calibrate_parameters(self):
        """
        校准模型参数。
        这个函数的目标是找出综合效益函数中β₀, β₁, β₂这三个参数的最佳值。
        这里使用逻辑回归模型来估计这些参数，这是一种简化的贝叶斯层次模型方法。
        """
        # 准备逻辑回归的特征（X）和目标（y）
        # 特征是孕妇BMI和GC含量变化
        X = self.male_data[['孕妇BMI', 'GC_change']].fillna(0)
        # 目标是Y染色体浓度是否达到4%的阈值（一个True/False的问题）
        y = self.male_data['Y_above_4pct']

        # 创建并训练逻辑回归模型
        lr = LogisticRegression(fit_intercept=True, random_state=42)
        lr.fit(X, y)

        # 从训练好的模型中提取参数
        beta_0 = lr.intercept_[0]  # 截距项
        beta_1, beta_2 = lr.coef_[0] # 系数项

        print(f"校准后的模型参数: β₀={beta_0:.4f}, β₁={beta_1:.4f}, β₂={beta_2:.4f}")

        return (beta_0, beta_1, beta_2)

    def optimize_detection_time(self, BMI, GC_change, beta_params):
        """
        为给定的BMI和GC值，找到最优的检测时间T。
        这本质上是一个寻找函数最大值的问题。
        """
        # 定义目标函数。因为优化工具通常是找最小值，所以我们对效益函数取负
        def objective(T):
            return -self.comprehensive_benefit_function(T, BMI, GC_change, beta_params)

        # 使用scipy的minimize_scalar在给定的临床窗口[12, 28]周内寻找最优解
        result = minimize_scalar(objective, bounds=(12, 28), method='bounded')

        optimal_T = result.x  # 最优的孕周T
        max_benefit = -result.fun # 最大的效益值（记得把负号加回来）

        return optimal_T, max_benefit

    def create_bmi_groups_and_optimize(self):
        """
        将所有样本按BMI分组，并为每个组找到最优的NIPT检测时间。
        """
        # 根据临床实践定义BMI分组的边界
        bmi_breaks = [20, 28, 32, 36, 40, 50]
        bmi_labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '[40,50)']

        # 首先，校准全局的beta参数
        beta_params = self.calibrate_parameters()

        results = [] # 用于存储每个组的分析结果

        # 遍历每个BMI分组
        for i, (lower, upper) in enumerate(zip(bmi_breaks[:-1], bmi_breaks[1:])):
            # 筛选出属于当前BMI组的数据
            group_data = self.male_data[
                (self.male_data['孕妇BMI'] >= lower) &
                (self.male_data['孕妇BMI'] < upper)
            ]

            if len(group_data) == 0: # 如果这个组没有样本，就跳过
                continue

            # 计算这个组的平均BMI和平均GC变化，作为该组的代表值
            avg_BMI = group_data['孕妇BMI'].mean()
            avg_GC_change = group_data['GC_change'].mean()

            # 为这个组找到最优的检测时间
            optimal_T, max_benefit = self.optimize_detection_time(avg_BMI, avg_GC_change, beta_params)

            # 计算与风险相关的指标
            risk_early = len(group_data[group_data['gestational_week'] < 12]) / len(group_data)
            risk_late = len(group_data[group_data['gestational_week'] > 22]) / len(group_data)

            # 将所有结果存入一个字典
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
        """
        分析检测时间误差对结果的影响。
        例如，如果推荐的最佳时间是12周，但实际检测是13周，这会带来多大的风险变化？
        """
        error_scenarios = [0, 0.5, 1.0, 1.5, 2.0]  # 假设的误差周数

        error_analysis = []

        for _, row in results_df.iterrows():
            base_week = row['Optimal_Week']
            base_risk = row['Total_Risk']

            for error in error_scenarios:
                # 模拟提前或推迟检测
                early_detection = max(12, base_week - error)
                late_detection = min(28, base_week + error)

                # 这里使用一个简化的模型来估计风险变化
                late_risk_change = error * 0.15  # 假设每推迟一周，风险增加15%

                error_analysis.append({
                    'BMI_Group': row['BMI_Group'],
                    'Error_Weeks': error,
                    'Risk_Change_Late': late_risk_change
                })

        return pd.DataFrame(error_analysis)

    def visualize_results(self, results_df, error_analysis_df):
        """
        创建一套完整的可视化图表来展示问题二的分析结果。
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('问题二：时间窗约束下的动态检测优化', fontsize=16)

        # 图1: 按BMI分组的最佳检测孕周
        ax1 = axes[0, 0]
        bars = ax1.bar(results_df['BMI_Group'], results_df['Optimal_Week'], color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('BMI分组')
        ax1.set_ylabel('最佳检测孕周')
        ax1.set_title('各BMI分组的最佳NIPT检测时间')
        ax1.set_ylim(12, 28) # Y轴范围与临床窗口一致
        for bar, week in zip(bars, results_df['Optimal_Week']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{week:.1f}', ha='center', va='bottom')

        # 图2: 按BMI分组的风险分析
        ax2 = axes[0, 1]
        x_pos = np.arange(len(results_df))
        width = 0.35
        ax2.bar(x_pos - width/2, results_df['Early_Risk'], width, label='过早检测风险', color='orange', alpha=0.7)
        ax2.bar(x_pos + width/2, results_df['Late_Risk'], width, label='过晚检测风险', color='red', alpha=0.7)
        ax2.set_xlabel('BMI分组')
        ax2.set_ylabel('风险比例')
        ax2.set_title('各BMI分组的风险分析')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(results_df['BMI_Group'], rotation=45)
        ax2.legend()

        # 图3: 检测时间误差的影响
        ax3 = axes[1, 0]
        for group in error_analysis_df['BMI_Group'].unique():
            group_data = error_analysis_df[error_analysis_df['BMI_Group'] == group]
            ax3.plot(group_data['Error_Weeks'], group_data['Risk_Change_Late'], 'o-', label=group, linewidth=2, markersize=6)
        ax3.set_xlabel('检测误差 (周数)')
        ax3.set_ylabel('风险变化量')
        ax3.set_title('检测时间误差对风险的影响')
        ax3.legend(title='BMI分组')
        ax3.grid(True, alpha=0.3)

        # 图4: 综合效益函数的可视化
        ax4 = axes[1, 1]
        T_range = np.linspace(12, 28, 100) # 生成一个从12到28的孕周序列
        beta_params = self.calibrate_parameters() # 重新校准参数以确保一致
        for bmi, color in zip([25, 30, 35, 40], ['blue', 'green', 'orange', 'red']):
            benefits = [self.comprehensive_benefit_function(t, bmi, 0, beta_params) for t in T_range]
            ax4.plot(T_range, benefits, color=color, label=f'BMI = {bmi}', linewidth=2)
        ax4.set_xlabel('孕周')
        ax4.set_ylabel('综合效益')
        ax4.set_title('不同BMI下的综合效益函数曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout() # 自动调整子图布局
        # 将最终的图表保存为高分辨率图片
        plt.savefig('new-plan/problem2_analysis.png', dpi=300, bbox_inches='tight')
        plt.show() # 在屏幕上显示图表

    def run_analysis(self):
        """
        运行问题二的完整分析流程。
        这是一个主方法，它会按顺序调用其他方法。
        """
        print("=== 问题二：时间窗约束下的动态检测优化分析开始 ===")

        # 第一步：按BMI分组并进行优化
        results_df, beta_params = self.create_bmi_groups_and_optimize()

        print("\n各BMI分组的最佳检测时间:")
        print("="*60)
        print(results_df[['BMI_Group', 'Optimal_Week', 'Sample_Size', 'Total_Risk']].round(3))

        print("\n校准后的模型参数:")
        print("="*40)
        print(f"β₀ = {beta_params[0]:.4f}, β₁ = {beta_params[1]:.4f}, β₂ = {beta_params[2]:.4f}")

        # 第二步：分析检测误差
        error_analysis_df = self.analyze_detection_errors(results_df)

        print("\n检测误差影响分析:")
        print("="*40)
        avg_error_impact = error_analysis_df.groupby('Error_Weeks')['Risk_Change_Late'].mean()
        for error, impact in avg_error_impact.items():
            print(f"误差 ±{error} 周: 平均风险变化 = +{impact:.3f}")

        # 第三步：生成并保存可视化结果
        self.visualize_results(results_df, error_analysis_df)

        # 第四步：保存详细的数值结果到CSV文件
        results_df.to_csv('new-plan/problem2_results.csv', index=False)
        error_analysis_df.to_csv('new-plan/problem2_error_analysis.csv', index=False)
        print("\n分析结果已保存到 problem2_results.csv 和 problem2_error_analysis.csv")

        return results_df, error_analysis_df

# 当这个脚本被直接运行时，执行以下代码
if __name__ == "__main__":
    # 创建问题二分析类的实例
    problem2 = Problem2_TimeWindowOptimization()
    # 运行完整的分析流程
    results_df, error_df = problem2.run_analysis()
