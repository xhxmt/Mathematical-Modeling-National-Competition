#!/usr/bin/env python3
"""
NIPT分析套件 - 完整执行脚本
这个脚本是整个分析项目的“总指挥”或“启动器”。
它的作用是按正确的顺序，依次调用和执行项目中的其他所有Python脚本，
从数据预处理开始，到各个问题的分析，最后到生成所有可视化图表。
"""

# 导入所有必需的库
import sys      # sys库用于与Python解释器交互，如此处的退出程序
import os       # os库用于与操作系统交互，如此处的检查文件是否存在
import subprocess # subprocess库是关键，它允许我们在Python脚本中执行外部命令，就像在终端中输入一样
import time     # time库用于计时
from datetime import datetime # datetime库用于获取当前时间

def run_analysis_suite():
    """
    运行完整的NIPT分析套件。
    这个主函数将协调所有分析步骤。
    """

    # --- 打印一个漂亮的标题，标志着分析开始 ---
    print("="*70)
    print("NIPT分析套件 - 完整实现")
    print("问题2, 3, 4的数学模型与求解")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # --- 步骤1: 检查并运行数据预处理 ---
    # 首先检查预处理后的数据文件是否存在。如果不存在，就先运行数据加载脚本。
    # 这是一个很好的健壮性设计，避免了不必要的重复工作。
    if not os.path.exists('new-plan/processed_data.csv'):
        print("未找到预处理数据，正在运行data_loader.py...")
        try:
            # 使用subprocess.run来执行'python3 data_loader.py'命令
            # check=True表示如果脚本执行出错，程序会抛出异常
            subprocess.run(['python3', 'new-plan/data_loader.py'], check=True)
            print("✓ 数据预处理完成")
        except subprocess.CalledProcessError as e:
            print(f"✗ 数据预处理失败: {e}")
            return False # 如果第一步就失败了，直接退出
    else:
        print("✓ 已找到预处理数据，跳过数据加载步骤。")

    # --- 步骤2: 定义要执行的分析脚本列表 ---
    # 将所有要执行的脚本和它们的描述放在一个列表中，方便后续循环处理
    analyses = [
        ("问题二分析", "new-plan/problem2_solution.py", "时间窗约束下的动态检测优化"),
        ("问题三分析", "new-plan/problem3_solution.py", "分层验证与动态风险量化"),
        ("问题四分析", "new-plan/problem4_solution.py", "动态多阶段检测优化"),
        ("高级Y染色体分析", "new-plan/advanced_y_chromosome_analysis.py", "独立的Y染色体浓度预测模型"),
        ("综合可视化", "new-plan/comprehensive_visualization.py", "生成所有总结性图表")
    ]

    results = {} # 用于存储每个脚本的执行结果

    # --- 步骤3: 循环执行所有分析脚本 ---
    for problem_name, script_name, description in analyses:
        print(f"\n{'-'*60}")
        print(f"正在运行: {problem_name} - {description}")
        print(f"脚本: {script_name}")

        start_time = time.time()

        try:
            # 再次使用subprocess.run来执行分析脚本
            # capture_output=True会捕获脚本的标准输出和错误信息
            subprocess.run(['python3', script_name], capture_output=True, text=True, check=True)

            execution_time = time.time() - start_time
            print(f"✓ {problem_name} 执行成功，耗时: {execution_time:.2f} 秒")
            results[problem_name] = {'status': '成功', 'execution_time': execution_time}

        except subprocess.CalledProcessError as e:
            # 如果脚本执行失败，则捕获错误并报告
            execution_time = time.time() - start_time
            print(f"✗ {problem_name} 执行失败，耗时: {execution_time:.2f} 秒")
            print(f"  错误信息: {e.stderr}") # 打印详细的错误日志
            results[problem_name] = {'status': '失败', 'execution_time': execution_time}

    # --- 步骤4: 生成最终的总结报告 ---
    print("\n" + "="*70)
    print("分析套件执行总结报告")
    print("="*70)

    total_time = sum(r['execution_time'] for r in results.values())
    successful_count = sum(1 for r in results.values() if r['status'] == '成功')

    print(f"总执行时间: {total_time:.2f} 秒")
    print(f"成功执行的分析: {successful_count}/{len(analyses)}")
    print()

    # --- 步骤5: 检查所有预期生成的文件是否存在 ---
    print("\n" + "="*70)
    print("生成的文件清单检查")
    print("="*70)

    output_files = [
        "new-plan/processed_data.csv",
        "new-plan/problem2_results.csv",
        "new-plan/problem2_error_analysis.csv",
        "new-plan/problem2_analysis.png",
        "new-plan/problem3_results.csv",
        "new-plan/problem3_analysis.png",
        "new-plan/problem4_results.csv",
        "new-plan/problem4_feature_importance.csv",
        "new-plan/problem4_analysis.png",
        "new-plan/advanced_y_chromosome_analysis.png",
        "new-plan/comprehensive_dashboard.png"
    ]

    for filename in output_files:
        if os.path.exists(filename):
            print(f"✓ {filename:45s} - 存在")
        else:
            print(f"✗ {filename:45s} - 缺失")

    print("\n" + "="*70)
    print(f"所有分析已于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 完成")
    print("="*70)

    # 如果有任何一步失败，整个过程就算失败
    return successful_count == len(analyses)

# 这是一个标准的Python写法，确保只有当这个脚本被直接执行时，下面的代码才会运行
if __name__ == "__main__":
    # 运行主分析函数
    success = run_analysis_suite()
    # 根据执行结果，向操作系统返回一个退出码
    # 退出码0通常表示成功，非0表示失败。这对于自动化脚本和工作流非常有用。
    sys.exit(0 if success else 1)
