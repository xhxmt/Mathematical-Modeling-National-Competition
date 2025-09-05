好的，这是您提供的文本的Markdown格式版本：

### 第三阶段数学模型：分层验证与动态风险量化模型

#### 1. 生存分析框架（Cox比例风险模型扩展）

引入时间依赖型协变量，构建动态风险函数：
$$h(t|X) = h_0(t) \exp[\beta_1 \cdot \text{BMI}(t) + \beta_2 \cdot \Delta\text{GC}(t) + \beta_3 \cdot Z(t)]$$
其中：
-   `BMI(t)` 为t孕周时标准化BMI值（动态更新）
-   `ΔGC(t) = GC(t) - GC(t-4)` 为过去四周GC含量变化率
-   `Z(t)` 为当前孕周Z值

参数估计采用部分似然函数：
$$L(\beta) = \prod_{i=1}^{n} \left[ \frac{\exp(\beta^T X_i(t_i))}{\sum_{j \in R(t_i)} \exp(\beta^T X_j(t_i))} \right]^{\delta_i}$$
其中 $\delta_i=1$ 表示该样本在 $t_i$ 孕周发生异常事件（如胎儿异常确诊）。

---

#### 2. 最优检测时间决策函数

在生存分析基础上，构建双目标优化问题：
$$\max_{T} [S(T) - \lambda \cdot \text{Risk}(T)]$$
其中：
-   **生存函数**：$S(T) = \exp\left[-\int_{0}^{T} h(u)du\right]$
-   **风险惩罚项**：$\text{Risk}(T) = \gamma_1 \cdot I(T<12) + \gamma_2 \cdot (T-16)^2 \cdot I(12 \le T \le 22) + \gamma_3 \cdot I(T>22)$

通过拉格朗日乘子法求解，得到最优时间条件：
$$\frac{\partial S(T)}{\partial T} = \lambda \cdot \frac{\partial \text{Risk}(T)}{\partial T}$$

---

#### 3. 分层交叉验证指标

定义**组间一致性指数 (Inter-group Consistency Index)**：
$$\text{ICI} = 1 - \frac{1}{k} \sum_{i=1}^{k} \frac{|\text{AUC}_i - \text{AUC}_{\text{pooled}}| + |\text{EARLY}_i - \text{EARLY}_{\text{pooled}}|}{\text{AUC}_{\text{pooled}} + \text{EARLY}_{\text{pooled}}}$$
其中：
-   $\text{AUC}_i$ 为第 $i$ 组的独立验证AUC值
-   $\text{EARLY}_i = \sum_{j \in G_i} I(T_{\text{pred},j} < T_{\text{true},j}) \cdot \exp(-0.1\Delta T_j)$

---

#### 4. 动态参数校准

采用贝叶斯层次模型进行参数更新：
$$\beta^{(m+1)} \sim N(\mu^{(m)}, \Sigma^{(m)})$$
其中均值向量更新公式：
$$\mu^{(m+1)} = \alpha \cdot \mu^{(m)} + (1-\alpha) \cdot \frac{1}{n_b} \sum_{b=1}^{n_b} \beta_b^{(m)}$$
$α=0.9$ 为记忆因子，$n_b=50$ 为Bootstrap抽样次数。