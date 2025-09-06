#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据探索脚本 - NIPT数据分析
探索附件.xlsx中的数据结构和内容
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """加载并探索Excel数据"""
    try:
        # 加载Excel文件
        df = pd.read_excel('附件.xlsx')
        print("数据加载成功！")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        print(df.head())
        
        print("\n数据类型:")
        print(df.dtypes)
        
        print("\n缺失值统计:")
        print(df.isnull().sum())
        
        print("\n基本统计信息:")
        print(df.describe())
        
        return df
    
    except Exception as e:
        print(f"数据加载出错: {e}")
        return None

if __name__ == "__main__":
    df = load_and_explore_data()