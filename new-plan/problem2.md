好的，这是您提供的文本的Markdown格式版本。

---

## 数学模型：时间窗约束下的动态检测时间优化模型

### 核心方程

定义综合效益函数：

$$\\Psi(T) = \\frac{1}{1 + e^{-(\\beta\_0 + \\beta\_1 \\text{BMI} + \\beta\_2 \\text{GC})}} \\times \\left\[1 - \\frac{T - 12}{22 - 12}\\right]^{w\_1} - \\lambda \\cdot \\left(\\frac{T}{40}\\right)^{w\_2} \\cdot I(T > 28)$$

其中：

* **第一项**：检测灵敏度（随孕周T呈逻辑增长）
* **第二项**：晚期检测风险惩罚项（$ \\lambda=0.85 $为临床验证系数）

### 约束条件

$$12 \\leq T \\leq 28 \\quad (\\text{临床可行时间窗})$$

$$\\frac{\\partial \\Psi}{\\partial T} = 0 \\quad (\\text{极值点存在性条件})$$

### 参数校准

采用贝叶斯层次模型：
$$\\beta\_i \\sim N(\\mu\_i, \\sigma\_i^2), \\quad \\mu\_i = \\alpha\_{i0} + \\alpha\_{i1}Z\_{\\text{score}}$$
**先验分布**：
$$w\_1 \\sim \\text{Gamma}(2, 0.5)$$
$$w\_2 \\sim \\text{Beta}(1.5, 2)$$

### 数值求解

使用自适应模拟退火算法：

```python
def cooling\_schedule(t):
    return 1000 / (1 + t\*\*0.8)

best\_T = simulated\_annealing(
    objective=psi\_function,
    bounds=(12, 28),
    cooling=cooling\_schedule,
    max\_iter=500
)
```

