
# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tabulate import tabulate
from pprint import pprint

math_score = pd.read_csv('raw student data/student-mat.csv')
portuguese_score = pd.read_csv('raw student data/student-por.csv')

# print("\n--- Column Names ---")
# print(*math_score.columns, sep=';')

col_data = [[i, col] for i, col in enumerate(math_score.columns)]
print("\nDataset Column Overview:")
print(tabulate(col_data, headers=["Index", "Column Name"], tablefmt="grid"))

# print("\n--- Pretty Printed Columns ---")
# print(math_score.columns)