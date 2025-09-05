# 导入所有必需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # 用于数据标准化
from sklearn.ensemble import RandomForestClassifier # 用于分类任务
from sklearn.metrics import classification_report, roc_auc_score # 用于评估分类模型
from sklearn.model_selection import train_test_split # 用于划分训练集和测试集
import warnings

# 忽略所有警告信息
warnings.filterwarnings('ignore')

# 设置matplotlib的全局字体和图形大小
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class Problem4_DynamicMultiStage:
    """
    这是一个为解决问题四（针对女性胎儿的动态多阶段检测优化）而设计的类。
    这个问题被构建成一个“动态规划”问题，可以想象成一个需要做出一系列决策的游戏。
    目标是：在整个孕期（12-28周）内，找到一个最佳的检测“策略”，以最大化长期收益。
    策略会告诉我们在每个时间点，根据孕妇的当前状况，是应该进行检测还是等待。
    """

    def __init__(self, data_path="/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv"):
        """
        类的构造函数。
        """
        self.df = pd.read_csv(data_path)
        self.T_max = 28  # 决策的最晚时间（孕周）
        self.T_min = 12  # 决策的最早时间（孕周）
        self.max_detections = 3  # 整个孕期最多允许进行3次检测
        self.setup_data()

    def setup_data(self):
        """
        为问题四准备和预处理数据。
        这里的关键是筛选出代表“女性胎儿”的代理样本，并定义与染色体异常相关的特征。
        """
        # --- 提取孕周数值 ---
        def extract_week(week_str):
            if pd.isna(week_str): return np.nan
            try: return float(str(week_str).split('w')[0])
            except: return np.nan
        self.df['gestational_week'] = self.df['检测孕周'].apply(extract_week)

        # --- 筛选女性胎儿代理样本 ---
        # 由于没有直接的女性胎儿标签，我们使用一个代理方法：
        # 将Y染色体浓度最低的25%的样本视为女性胎儿（因为理论上女性胎儿Y染色体浓度应为0）。
        y_threshold = self.df['Y_concentration'].quantile(0.25)

        self.female_data = self.df[
            (self.df['Y_concentration'] <= y_threshold) &
            (self.df['gestational_week'].notna()) &
            (self.df['gestational_week'].between(12, 28)) &
            (self.df['孕妇BMI'].notna()) &
            (self.df['GC含量'].notna()) &
            (self.df['X染色体的Z值'].notna()) &
            (self.df['21号染色体的Z值'].notna()) &
            (self.df['18号染色体的Z值'].notna()) &
            (self.df['13号染色体的Z值'].notna())
        ].copy()

        # --- 定义“异常”事件 ---
        # 使用真实的“胎儿是否健康”列作为我们最终要预测的目标
        self.female_data['is_abnormal'] = (self.female_data['胎儿是否健康'] == '否')
        # 同时，也根据各染色体的Z值（一个风险分数）来定义异常
        z_threshold = 2.0  # Z值大于2通常被认为有风险
        self.female_data['z_abnormal'] = (
            (np.abs(self.female_data['21号染色体的Z值']) > z_threshold) |
            (np.abs(self.female_data['18号染色体的Z值']) > z_threshold) |
            (np.abs(self.female_data['13号染色体的Z值']) > z_threshold) |
            (np.abs(self.female_data['X染色体的Z值']) > z_threshold)
        )

        if len(self.female_data) > 0:
            # 标准化所有数值特征，使得它们在同一尺度上，方便模型处理
            features_to_scale = ['孕妇BMI', 'GC含量', '21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值', 'X染色体的Z值']
            self.female_data[features_to_scale] = StandardScaler().fit_transform(self.female_data[features_to_scale])

            # 计算GC含量变化率
            self.female_data = self.female_data.sort_values(['孕妇代码', 'gestational_week'])
            self.female_data['Delta_GC'] = self.female_data.groupby('孕妇代码')['GC含量'].diff().fillna(0)

        print(f"用于问题四分析的女性胎儿代理样本数: {len(self.female_data)}")
        if len(self.female_data) > 0:
            print(f"真实异常率: {self.female_data['is_abnormal'].mean()*100:.2f}%")
            print(f"基于Z值的异常率: {self.female_data['z_abnormal'].mean()*100:.2f}%")

    def reward_function(self, state, action, detection_history):
        """
        定义“收益函数” R(s, a)。
        这个函数告诉我们，在某个状态s下，采取某个行动a能得到多少“分数”（收益）。
        收益可以是正的（如成功检测到异常），也可以是负的（如检测成本、风险）。
        R(s_t, a_t) = α·Sen(s_t)·a_t - β·Risk(t)·a_t
        """
        alpha, beta = 0.7, 0.3 # 收益和风险的权重
        bmi_t, gc_change, z_t, t = state

        if action == 1:  # 如果行动是“检测”
            # 收益部分：检测的灵敏度（检测到问题的能力）
            sensitivity = 1 / (1 + np.exp(-(0.8 * z_t + 0.2 * gc_change)))

            # 风险/成本部分
            risk_1 = 0.1 * max(0, t - 22) ** 2 # 过晚检测的风险
            risk_2 = 0.05 * sum(detection_history) # 重复检测的成本
            total_risk = risk_1 + risk_2

            return alpha * sensitivity - beta * total_risk
        else:  # 如果行动是“不检测”，则没有收益也没有成本
            return 0

    def state_transition(self, current_state, action, noise_std=0.1):
        """
        定义“状态转移函数”。
        这个函数描述了世界如何演变。如果我们今天在状态s_t，并采取了行动a_t，
        那么明天我们会到达哪个新状态s_{t+1}？
        这里我们用一个简化的模型来模拟孕妇指标的自然变化。
        """
        bmi_t, gc_change, z_t, t = current_state

        # BMI、GC、Z值都会有一些小的随机波动
        bmi_next = bmi_t + np.random.normal(0, noise_std)
        gc_next = 0.7 * gc_change + np.random.normal(0, noise_std) # GC变化具有一定的自相关性
        z_next = z_t + np.random.normal(0, noise_std * 0.5)
        t_next = t + 1 # 时间总是向前流逝

        return np.array([bmi_next, gc_next, z_next, t_next])

    def dynamic_programming_solver(self):
        """
        动态规划求解器，使用“反向归纳法”来求解贝尔曼方程。
        核心思想：从最后一天（第28周）开始倒着往回推算。
        在第28周，我们知道游戏结束了，所以未来的价值是0。
        然后我们推算第27周的最佳决策：比较“检测”和“不检测”哪个总收益（当前收益 + 未来期望收益）更高。
        这样一步步倒推到第12周，我们就能得到每个时间点、每种状态下的最优决策。
        V_t(s_t) = max_{a_t} { R(s_t,a_t) + E[V_{t+1}(s_{t+1})] }
        """
        # V(t, state_idx, detections_done) -> value
        value_functions = {}
        # P(t, state_idx, detections_done) -> action
        optimal_policies = {}

        # 创建一组有代表性的状态，用于计算
        sample_states = self.female_data[['孕妇BMI', 'Delta_GC', '21号染色体的Z值']].values

        print("开始求解动态规划模型...")
        # 从最后一天开始反向循环
        for t in range(self.T_max, self.T_min - 1, -1):
            for state_idx in range(len(sample_states)):
                for detections_done in range(self.max_detections + 1):
                    state = np.append(sample_states[state_idx], t)

                    # 终止条件：在最后一天，未来价值为0
                    if t == self.T_max:
                        value_functions[(t, state_idx, detections_done)] = 0
                        optimal_policies[(t, state_idx, detections_done)] = 0 # 结束时不检测
                        continue

                    # --- 对于每个状态，比较采取不同行动的价值 ---
                    # 价值(不检测)
                    reward_wait = self.reward_function(state, 0, [0]*detections_done)
                    future_value_wait = value_functions.get((t + 1, state_idx, detections_done), 0)
                    value_wait = reward_wait + future_value_wait

                    # 价值(检测)
                    if detections_done < self.max_detections:
                        reward_detect = self.reward_function(state, 1, [0]*detections_done)
                        future_value_detect = value_functions.get((t + 1, state_idx, detections_done + 1), 0)
                        value_detect = reward_detect + future_value_detect
                    else:
                        value_detect = -np.inf # 如果已达最大检测次数，则不能再检测

                    # --- 做出最优决策 ---
                    if value_detect > value_wait:
                        value_functions[(t, state_idx, detections_done)] = value_detect
                        optimal_policies[(t, state_idx, detections_done)] = 1 # 最优行动是检测
                    else:
                        value_functions[(t, state_idx, detections_done)] = value_wait
                        optimal_policies[(t, state_idx, detections_done)] = 0 # 最优行动是不检测

        print("求解完成。")
        return value_functions, optimal_policies

    def analyze_optimal_detection_strategy(self, optimal_policies):
        """根据求解得到的最优策略，进行分析。"""
        detection_recommendations = []
        # 分析在平均状态下，且尚未进行任何检测时的策略
        avg_state_idx = len(self.female_data) // 2

        for t in range(self.T_min, self.T_max):
            action = optimal_policies.get((t, avg_state_idx, 0), 0) # 假设从0次检测开始
            detection_recommendations.append({
                'week': t,
                'recommended_action': 'Detect' if action == 1 else 'Wait'
            })
        return detection_recommendations

    def abnormality_classification(self):
        """
        训练一个分类模型，用来识别哪些因素对判断胎儿异常最重要。
        这里使用随机森林模型，因为它能很好地处理复杂的非线性关系，并给出特征重要性。
        """
        features = ['孕妇BMI', 'GC含量', 'X染色体的Z值', '21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值', 'gestational_week']
        X = self.female_data[features]
        y = self.female_data['is_abnormal'] # 真实健康状况

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)

        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        print("\n分类模型性能:")
        print(f"AUC 分数: {auc:.4f}")

        feature_importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)

        return feature_importance, auc

    def visualize_results(self, detection_recommendations, feature_importance):
        """为问题四创建一套完整的可视化图表。"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('问题四：动态多阶段检测优化', fontsize=16)

        # 图1: 最优检测策略
        ax1 = axes[0]
        weeks = [rec['week'] for rec in detection_recommendations]
        actions = [1 if rec['recommended_action'] == 'Detect' else 0 for rec in detection_recommendations]
        ax1.step(weeks, actions, where='post', color='coral', linewidth=3)
        ax1.set_xlabel('孕周')
        ax1.set_ylabel('推荐行动 (1=检测, 0=等待)')
        ax1.set_title('平均状态下的最优检测策略')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Wait', 'Detect'])
        ax1.grid(True, alpha=0.3)

        # 图2: 异常检测的特征重要性
        ax2 = axes[1]
        ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue', edgecolor='black')
        ax2.set_xlabel('特征重要性')
        ax2.set_title('识别胎儿异常的特征重要性')
        ax2.invert_yaxis()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('new-plan/problem4_analysis.png', dpi=300)
        plt.show()

    def run_analysis(self):
        """运行问题四的完整分析流程。"""
        print("=== 问题四：动态多阶段检测优化分析开始 ===")

        value_functions, optimal_policies = self.dynamic_programming_solver()
        detection_recommendations = self.analyze_optimal_detection_strategy(optimal_policies)

        print("\n最优检测策略 (平均状态):")
        print(pd.DataFrame(detection_recommendations))

        feature_importance, auc = self.abnormality_classification()
        print("\n特征重要性排名:")
        print(feature_importance)

        self.visualize_results(detection_recommendations, feature_importance)

        # 保存结果
        pd.DataFrame(detection_recommendations).to_csv('new-plan/problem4_results.csv', index=False)
        feature_importance.to_csv('new-plan/problem4_feature_importance.csv', index=False)
        print("\n分析结果已保存到 problem4_results.csv 和 problem4_feature_importance.csv")

if __name__ == "__main__":
    problem4 = Problem4_DynamicMultiStage()
    problem4.run_analysis()
