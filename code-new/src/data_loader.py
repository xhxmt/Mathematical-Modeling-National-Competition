import pandas as pd
import numpy as np
import os

def clean_column_names(df):
    """Removes leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df

def parse_gestational_week(week_str):
    """Converts gestational week string to a numerical float."""
    if pd.isna(week_str):
        return np.nan
    week_str = str(week_str)
    try:
        if 'w' in week_str.lower():
            parts = week_str.lower().split('w')
            weeks = int(parts[0])
            days = 0
            if len(parts) > 1 and parts[1]:
                day_part = parts[1].replace('+', '').strip()
                if day_part:
                    days = int(day_part)
            return weeks + days / 7.0
        else:
            return float(week_str)
    except (ValueError, TypeError):
        return np.nan

def get_cleaned_data():
    """
    Loads, cleans, and preprocesses the NIPT data from the Excel file.
    Reads from the root directory.
    """
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', '附件.xlsx')

    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data.xlsx')
        if not os.path.exists(file_path):
             raise FileNotFoundError("Could not find the data file '附件.xlsx' or 'data.xlsx' in the root directory.")

    male_df = pd.read_excel(file_path, sheet_name='男胎检测数据')
    female_df = pd.read_excel(file_path, sheet_name='女胎检测数据')

    # --- Preprocess Male Data ---
    male_df = clean_column_names(male_df)
    male_df['Gestational_Week'] = male_df['检测孕周'].apply(parse_gestational_week)
    male_df.rename(columns={
        '孕妇BMI': 'BMI',
        'Y染色体浓度': 'Y_Concentration',
        '年龄': 'Age',
        '身高': 'Height',
        '体重': 'Weight'
    }, inplace=True)
    male_df.dropna(subset=['Gestational_Week', 'BMI', 'Y_Concentration', 'Age', 'Height', 'Weight'], inplace=True)

    # --- Preprocess Female Data ---
    female_df = clean_column_names(female_df)
    female_df['Gestational_Week'] = female_df['检测孕周'].apply(parse_gestational_week)
    female_df['Is_Abnormal'] = female_df['染色体的非整倍体'].notna().astype(int)
    female_df.rename(columns={
        '孕妇BMI': 'BMI',
        '年龄': 'Age',
        '13号染色体的Z值': 'Z_Score_13',
        '18号染色体的Z值': 'Z_Score_18',
        '21号染色体的Z值': 'Z_Score_21',
        'X染色体的Z值': 'Z_Score_X',
        'X染色体浓度': 'X_Concentration',
        'GC含量': 'GC_Content'
    }, inplace=True)
    female_df = female_df.loc[:, ~female_df.columns.str.contains('^Unnamed')]

    return male_df, female_df

if __name__ == '__main__':
    try:
        male_data, female_data = get_cleaned_data()
        print("Data loaded and preprocessed successfully.")
        print("\n--- Male Data Info ---")
        print(male_data.info())
        print("\n--- Female Data Info ---")
        print(female_data.info())
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
