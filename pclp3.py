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

# TASK 2

survived_rate = df['Survived'].mean() * 100
not_survived_rate = 100 - survived_rate
class_counts = df['Pclass'].value_counts(normalize=True) * 100
gender_counts = df['Sex'].value_counts(normalize=True) * 100
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs[0, 0].bar(['Survived', 'Not Survived'], [survived_rate, not_survived_rate], color=['blue', 'red'])
axs[0, 0].set_title('Survival Rates')
axs[0, 0].set_ylabel('Percentage')
class_counts.plot(kind='bar', ax=axs[0, 1], color='green')
axs[0, 1].set_title('Passenger Class Distribution')
axs[0, 1].set_ylabel('Percentage')
gender_counts.plot(kind='bar', ax=axs[1, 0], color='purple')
axs[1, 0].set_title('Gender Distribution')
axs[1, 0].set_ylabel('Percentage')
plt.tight_layout()

plt.show()

#TASK 3

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
numeric_columns = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
for i, col in enumerate(numeric_columns):
    ax = axs[i//3, i%3]
    df[col].hist(ax=ax, bins=10, color='skyblue', edgecolor='black')
    ax.set_title('Histogram for {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

#TASK 4

missing_data = df.isnull().sum()
missing_columns = missing_data[missing_data > 0]
missing_proportions = (missing_columns / len(df)) * 100
print("Coloumns with missing values:")
print(missing_proportions)
missing_by_survival = df.groupby('Survived').apply(lambda x: x.isnull().sum() / len(x) * 100)
print("\nThe procent for every missing value:")
print(missing_by_survival[missing_columns.index])