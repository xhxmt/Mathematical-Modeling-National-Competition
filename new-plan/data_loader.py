# 导入所有必需的库
import pandas as pd  # Pandas是用于数据处理和分析的核心库，我们将用它来读取和操作数据
import numpy as np  # Numpy是用于科学计算的库，特别是在处理数值数组时非常有用
import matplotlib.pyplot as plt  # Matplotlib是用于创建图表和可视化的主要库
import seaborn as sns  # Seaborn是基于Matplotlib的更高级的库，可以创建更美观的统计图表
from datetime import datetime  # Datetime库用于处理日期和时间
import warnings  # Warnings库用于控制警告信息的显示

# 设置全局参数，以优化后续的分析和可视化
# 忽略所有警告信息，这有助于保持输出的整洁，但在调试时可能需要关闭
warnings.filterwarnings('ignore')

# 设置matplotlib的全局字体和图形大小，以确保图表清晰可读
# 'DejaVu Sans' 是一种支持多种字符的字体，可以避免乱码问题
plt.rcParams['font.family'] = 'DejaVu Sans'
# 设置默认字体大小
plt.rcParams['font.size'] = 10
# 设置默认图形的尺寸（12英寸宽，8英寸高）
plt.rcParams['figure.figsize'] = (12, 8)

def load_nipt_data(file_path):
    """
    这个函数负责从指定的Excel文件中加载原始NIPT（无创产前检测）数据。

    参数:
    file_path (str): 包含原始数据的Excel文件的路径。

    返回:
    pandas.DataFrame: 一个包含加载数据的DataFrame，如果加载失败则返回None。
    """
    try:
        # 使用pandas的read_excel函数来读取文件内容
        # 这是处理表格数据的第一步
        df = pd.read_excel(file_path)

        # 打印一些基本信息来验证数据是否被正确加载
        print(f"成功加载数据集，其维度（行数, 列数）为: {df.shape}")
        print(f"数据集包含的列名: {list(df.columns)}")

        # 显示数据的前5行，这有助于我们快速了解数据的结构和内容
        print("\n数据前5行预览:")
        print(df.head())

        # 打印每列的数据类型，以检查是否有需要转换格式的列
        print("\n各列的数据类型:")
        print(df.dtypes)

        # 检查每列有多少缺失值，这对于后续的数据清洗至关重要
        print("\n各列的缺失值统计:")
        print(df.isnull().sum())

        # 如果一切顺利，返回加载好的数据
        return df
    except Exception as e:
        # 如果在加载过程中出现任何错误（例如文件不存在、格式错误），则打印错误信息
        print(f"加载数据时发生错误: {e}")
        # 并返回None，表示加载失败
        return None

def preprocess_data(df):
    """
    这个函数负责对加载的原始数据进行预处理和清洗，以便进行后续的数学建模分析。

    参数:
    df (pandas.DataFrame): 从load_nipt_data函数获得的原始数据。

    返回:
    pandas.DataFrame: 经过处理和特征工程后的数据，如果输入为空则返回None。
    """
    # 首先检查输入的数据是否有效
    if df is None:
        return None

    # 创建一个原始数据的副本，这是一个好习惯，可以避免在处理过程中意外修改了原始数据
    processed_df = df.copy()

    # --- 特征工程：计算BMI ---
    # BMI（身体质量指数）是一个重要的健康指标，可能与NIPT检测结果相关
    # 我们需要从列名中自动识别身高和体重的列
    # 这里通过关键词（如'height', '身高', 'weight', '体重'）来动态查找列名
    height_cols = [col for col in df.columns if 'height' in col.lower() or '身高' in str(col) or 'D' in str(col)]
    weight_cols = [col for col in df.columns if 'weight' in col.lower() or '体重' in str(col) or 'E' in str(col)]

    # 如果同时找到了身高和体重的列
    if height_cols and weight_cols:
        height_col = height_cols[0]
        weight_col = weight_cols[0]

        # BMI的计算公式是：体重(kg) / (身高(m))^2
        # 注意原始数据中身高单位可能是cm，所以需要除以100转换为m
        processed_df['BMI'] = processed_df[weight_col] / (processed_df[height_col] / 100) ** 2

        # 打印BMI的描述性统计信息（如平均值、标准差等），以了解其分布情况
        print(f"\nBMI指标的统计信息:")
        print(processed_df['BMI'].describe())

    # --- 特征工程：分析Y染色体浓度 ---
    # Y染色体浓度是判断胎儿性别的关键。通常，只有男性胎儿才会检测到Y染色体。
    # 自动识别Y染色体浓度的列
    y_conc_cols = [col for col in df.columns if 'Y' in str(col) and ('浓度' in str(col) or 'conc' in str(col).lower())]
    if y_conc_cols:
        y_conc_col = y_conc_cols[0]
        # 将该列转换为数值类型，对于无法转换的值（如文本），则将其设为缺失值（NaN）
        processed_df['Y_concentration'] = pd.to_numeric(processed_df[y_conc_col], errors='coerce')

        # --- 数据筛选：提取男性胎儿样本 ---
        # 我们将有Y染色体浓度数据的样本视为男性胎儿样本
        # .notna() 会筛选出所有非缺失值的行
        male_data = processed_df[processed_df['Y_concentration'].notna()].copy()

        print(f"\n男性胎儿样本的Y染色体浓度统计信息:")
        print(male_data['Y_concentration'].describe())

        # --- 创建一个新特征：Y染色体浓度是否达标 ---
        # 在NIPT检测中，Y染色体浓度达到某个阈值（例如4%）才被认为是可靠的
        # 这里我们创建一个布尔型（True/False）的列来标记是否达标
        male_data['Y_above_4pct'] = male_data['Y_concentration'] >= 4.0
        # 计算并打印达标样本的百分比
        print(f"\nY染色体浓度达到或超过4%的男性胎儿样本占比: {male_data['Y_above_4pct'].mean()*100:.2f}%")

    # 返回经过完整预处理的数据
    return processed_df

# 这是一个标准的Python写法，确保只有当这个脚本被直接执行时，下面的代码才会运行
# 如果这个脚本被其他脚本导入（import），下面的代码则不会执行
if __name__ == "__main__":
    # --- 主执行流程 ---

    # 定义原始数据文件的路径
    # 注意：这是一个硬编码的本地路径，在不同的电脑上运行时可能需要修改
    file_path = "/home/tfisher/code/math/2025/c-problem/附件.xlsx"

    # 第一步：调用函数加载数据
    df = load_nipt_data(file_path)

    # 检查数据是否成功加载
    if df is not None:
        # 第二步：如果加载成功，则调用函数进行数据预处理
        processed_df = preprocess_data(df)

        # 第三步：将处理好的数据保存到新的文件中
        # 这是一个非常重要的步骤，因为预处理可能很耗时，
        # 将结果保存下来，后续的分析就可以直接读取这个文件，而无需重复预处理
        if processed_df is not None:
            # 使用.to_csv()方法将DataFrame保存为CSV文件
            # index=False表示我们不希望在文件中保存DataFrame的索引号
            processed_df.to_csv("new-plan/processed_data.csv", index=False)
            print("\n处理完成的数据已成功保存到 new-plan/processed_data.csv")
