import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
#TASK 1

df = pd.read_csv('train.csv')
num_columns = df.shape[1]
print("Number of columns:", num_columns)
print("\nData type for every coloumn:")
print(df.dtypes)
print("\nNumber of absent values for every column:")
print(df.isnull().sum())
num_rows = df.shape[0]
print("\nThe total number of lines:", num_rows)
duplicates = df.duplicated().any()
print("\nThere are duplicated lines:", duplicates)