import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a font that supports both English and better rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

def load_nipt_data(file_path):
    """Load and preprocess NIPT data from Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display first few rows to understand the structure
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the NIPT data for analysis"""
    if df is None:
        return None
    
    # Make a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Calculate BMI if height and weight columns exist
    height_cols = [col for col in df.columns if 'height' in col.lower() or '身高' in str(col) or 'D' in str(col)]
    weight_cols = [col for col in df.columns if 'weight' in col.lower() or '体重' in str(col) or 'E' in str(col)]
    
    if height_cols and weight_cols:
        height_col = height_cols[0]
        weight_col = weight_cols[0]
        
        # Calculate BMI = weight(kg) / (height(m))^2
        processed_df['BMI'] = processed_df[weight_col] / (processed_df[height_col] / 100) ** 2
        
        print(f"\nBMI statistics:")
        print(processed_df['BMI'].describe())
    
    # Identify Y chromosome concentration column
    y_conc_cols = [col for col in df.columns if 'Y' in str(col) and ('浓度' in str(col) or 'conc' in str(col).lower())]
    if y_conc_cols:
        y_conc_col = y_conc_cols[0]
        processed_df['Y_concentration'] = pd.to_numeric(processed_df[y_conc_col], errors='coerce')
        
        # Filter for male fetuses (those with Y chromosome data)
        male_data = processed_df[processed_df['Y_concentration'].notna()].copy()
        
        print(f"\nY chromosome concentration statistics (male fetuses):")
        print(male_data['Y_concentration'].describe())
        
        # Identify if concentration reaches 4% threshold
        male_data['Y_above_4pct'] = male_data['Y_concentration'] >= 4.0
        print(f"\nPercentage of male fetuses with Y concentration >= 4%: {male_data['Y_above_4pct'].mean()*100:.2f}%")
    
    return processed_df

if __name__ == "__main__":
    # Load and preprocess data
    file_path = "/home/tfisher/code/math/2025/c-problem/附件.xlsx"
    df = load_nipt_data(file_path)
    
    if df is not None:
        processed_df = preprocess_data(df)
        
        # Save processed data for further analysis
        if processed_df is not None:
            processed_df.to_csv("/home/tfisher/code/math/2025/c-problem/new-plan/processed_data.csv", index=False)
            print("\nProcessed data saved to processed_data.csv")