# 导入所有必需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec # 一个更灵活的工具，用于创建复杂的子图布局
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# --- 全局绘图风格设置 ---
# 设置matplotlib的默认字体和样式，以确保所有图表的美观和一致性
plt.rcParams['font.family'] = 'DejaVu Sans' # 使用支持多种字符的字体
plt.rcParams['font.size'] = 12 # 设置稍大一点的默认字体大小
plt.rcParams['figure.figsize'] = (18, 14) # 设置默认的图形尺寸
plt.style.use('seaborn-v0_8-whitegrid') # 使用一个带网格的清爽样式

class ComprehensiveVisualization:
    """
    这是一个总的可视化类，它的任务是读取之前所有问题分析生成的结果，
    并将这些结果汇总成一系列清晰、全面的图表和仪表盘。
    可以把它看作是整个项目的“报告生成器”。
    """

    def __init__(self):
        """
        类的构造函数。
        """
        # 初始化时，直接调用方法加载所有需要的数据
        self.load_results()

    def load_results(self):
        """
        从各个CSV文件中加载之前步骤生成的分析结果。
        """
        try:
            # 加载问题二的结果
            self.problem2_results = pd.read_csv('new-plan/problem2_results.csv')
            self.problem2_errors = pd.read_csv('new-plan/problem2_error_analysis.csv')

            # 加载问题三的结果
            self.problem3_results = pd.read_csv('new-plan/problem3_results.csv')

            # 加载问题四的结果
            self.problem4_results = pd.read_csv('new-plan/problem4_results.csv')
            self.problem4_features = pd.read_csv('new-plan/problem4_feature_importance.csv')

            # 加载最开始经过预处理的完整数据集
            self.processed_data = pd.read_csv('new-plan/processed_data.csv')

            print("所有分析结果已成功加载！")

        except Exception as e:
            print(f"加载结果时发生错误: {e}")
            print("请确保问题2, 3, 4的分析脚本都已成功运行，并生成了对应的.csv结果文件。")

    def create_overview_dashboard(self):
        """
        创建一张最核心的、包含所有问题关键结果的“概览仪表盘”。
        这张图利用GridSpec将多个子图拼接在一起，形成一个信息丰富的单一视图。
        """
        # 创建一个20x16英寸的大画布，并使用GridSpec定义一个4x4的网格布局
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        fig.suptitle('NIPT分析：综合结果仪表盘', fontsize=24, fontweight='bold')

        # --- 将各个子图放置在网格的不同位置 ---
        # 问题二的结果
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_problem2_optimal_timing(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_problem2_error_impact(ax2)

        # 问题三的结果
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_problem3_parameters(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_problem3_validation(ax4)

        # 问题四的结果
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_problem4_detection_strategy(ax5)

        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_problem4_feature_importance(ax6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('new-plan/comprehensive_dashboard.png', dpi=300)
        plt.show()

    # --- 为每个问题结果设计的独立绘图函数 ---
    # 每个函数都只负责绘制一个特定的子图，这使得代码更加模块化和易于管理。

    def plot_problem2_optimal_timing(self, ax):
        """绘制问题二的结果：不同BMI分组的最佳检测时间。"""
        ax.bar(self.problem2_results['BMI_Group'], self.problem2_results['Optimal_Week'], color='steelblue', alpha=0.8)
        ax.set_title('问题二: 各BMI分组的最佳检测时间', fontsize=14, fontweight='bold')
        ax.set_ylabel('最佳孕周')
        ax.set_ylim(11, 15)
        ax.grid(True, which='major', linestyle='--', alpha=0.6)

    def plot_problem2_error_impact(self, ax):
        """绘制问题二的结果：检测时间误差带来的风险变化。"""
        error_impact = self.problem2_errors.groupby('Error_Weeks')['Risk_Change_Late'].mean()
        ax.plot(error_impact.index, error_impact.values, 'ro-', linewidth=2, markersize=8)
        ax.set_title('问题二: 检测时间误差的影响', fontsize=14, fontweight='bold')
        ax.set_xlabel('检测误差 (周数)')
        ax.set_ylabel('平均风险变化')

    def plot_problem3_parameters(self, ax):
        """绘制问题三的结果：模型参数在贝叶斯更新前后的对比。"""
        param_names = self.problem3_results['Parameter']
        original = self.problem3_results['Original_Value']
        updated = self.problem3_results['Updated_Value']
        x = np.arange(len(param_names))
        ax.bar(x - 0.2, original, 0.4, label='原始参数', color='orange')
        ax.bar(x + 0.2, updated, 0.4, label='更新后参数', color='purple')
        ax.set_title('问题三: 参数更新前后对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()

    def plot_problem3_validation(self, ax):
        """绘制问题三的结果：交叉验证的性能指标。"""
        metrics = ['AUC Score', 'EARLY Score', 'ICI Score']
        values = [0.5912, 0.2461, 0.9417] # 从分析报告中获取的硬编码值
        ax.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax.set_title('问题三: 交叉验证性能', fontsize=14, fontweight='bold')
        ax.set_ylabel('分数')
        ax.set_ylim(0, 1.1)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    def plot_problem4_detection_strategy(self, ax):
        """绘制问题四的结果：动态规划给出的最优检测策略。"""
        ax.bar(self.problem4_results['week'], self.problem4_results['detection_rate'], color='coral', alpha=0.8)
        ax.axhline(y=0.5, color='red', linestyle='--', label='决策阈值')
        ax.set_title('问题四: 动态检测策略', fontsize=14, fontweight='bold')
        ax.set_xlabel('孕周')
        ax.set_ylabel('检测概率')
        ax.legend()

    def plot_problem4_feature_importance(self, ax):
        """绘制问题四的结果：用于识别异常的最重要特征。"""
        top_features = self.problem4_features.head(6)
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, palette='viridis')
        ax.set_title('问题四: 识别异常的关键特征', fontsize=14, fontweight='bold')
        ax.set_xlabel('特征重要性')

    def run_visualization_suite(self):
        """
        运行完整的可视化套件。
        这将生成并保存所有定义好的图表。
        """
        print("开始创建综合可视化图表...")

        # 1. 创建并保存主仪表盘
        print("正在生成主仪表盘 (comprehensive_dashboard.png)...")
        self.create_overview_dashboard()

        # 这里可以添加调用其他绘图函数（如为每个问题单独生成总结图）的代码
        # self.create_individual_problem_summaries()
        # self.create_methodology_comparison()

        print("\n所有可视化任务完成！")

# 当这个脚本被直接运行时，执行以下代码
if __name__ == "__main__":
    # 创建可视化类的实例
    visualizer = ComprehensiveVisualization()
    # 运行完整的可视化流程
    visualizer.run_visualization_suite()
