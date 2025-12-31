
# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1. Strip whitespace and replace spaces with underscores
# 2. Convert to lowercase
# 3. replace the spaces with underscores
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df



# check for missing values
def check_missing_data(df_path):
    df = pd.read_csv(df_path , sep=';')
    print("\n" + "="*30)
    print("MISSING VALUES REPORT")
    print("="*30)
    
    # 1. Check Missing Values per Column
    # .sum() treats True as 1 and False as 0
    missing_cols = df.isnull().sum()
    
    print("\n--- By Column ---")
    if missing_cols.sum() == 0:
        print("Clean! No missing values found in any column.")
    else:
        # Only show columns that actually have missing data
        print(missing_cols[missing_cols > 0])
        
    # 2. Check Missing Values per Row (Total rows with at least one missing)
    # axis=1 checks across the row
    missing_rows_count = df.isnull().any(axis=1).sum()
    
    print("\n--- By Row ---")
    print(f"Total rows with missing data: {missing_rows_count} out of {len(df)}")
    print("="*30 + "\n")



def split_data(X, y, test_size=0.15, val_size= 0.15 , random_state=42):
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state , shuffle=True)
    
    val_size_relative = val_size / (1 - test_size)
    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_relative, random_state=random_state ,shuffle=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test



def print_split_shapes(name, X_train, X_val, X_test, y_train, y_val, y_test):
    print(f"\n{name} dataset shapes")
    print("-" * 40)
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")



def load_and_split_data(file_path, target_column='g3', test_size=0.15, val_size=0.15 , random_state=42):
    # Load data
    df = pd.read_csv(file_path, sep=';')
    
    # Clean column names
    df = clean_column_names(df)
    
    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split data
    return split_data(X, y, test_size, val_size, random_state)


if __name__ == "__main__":
    math_file_path = Path('data/student-mat.csv')
    portuguese_file_path = Path('data/student-por.csv')
    
    X_train_math, X_val_math, X_test_math, y_train_math, y_val_math, y_test_math = load_and_split_data(math_file_path)
    X_train_port, X_val_port, X_test_port, y_train_port, y_val_port, y_test_port = load_and_split_data(portuguese_file_path)
    
    print_split_shapes(
    "Math",X_train_math, X_val_math, X_test_math, 
    y_train_math, y_val_math, y_test_math
    )
    
    print_split_shapes(
    "Portuguese",
    X_train_port, X_val_port, X_test_port,
    y_train_port, y_val_port, y_test_port
    )