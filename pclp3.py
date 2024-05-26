import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

# TASK 1

df = pd.read_csv('train.csv')
# Aflu numarul de coloane
num_columns = df.shape[1]
print("Number of columns:", num_columns)
print("\nData type for every coloumn:")
# Afisez tipurile pentru fiecare coloana
print(df.dtypes)
print("\nNumber of absent values for every column:")
# Determin numarul de valori lipsa pe coloana
print(df.isnull().sum())
# Aflu umarul de linii
num_rows = df.shape[0]
print("\nThe total number of lines:", num_rows)
# Aflu numarul de linii duplicate
duplicates = df.duplicated().any()
print("\nThere are duplicated lines:", duplicates)

# TASK 2

# Calculez rata de supravietuire
survived_rate = df['Survived'].mean() * 100
# Calculez complementara pentru rata de supravietuire
not_survived_rate = 100 - survived_rate
# Calculez procentele pentru clasa pasagerilor si gen
class_counts = df['Pclass'].value_counts(normalize=True) * 100
gender_counts = df['Sex'].value_counts(normalize=True) * 100
# Creez figura si setez titlurile label-urilor
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# Graficele pentru ratele de supravietuire
axs[0, 0].bar(['Survived', 'Not Survived'], [survived_rate, not_survived_rate], color=['blue', 'red'])
axs[0, 0].set_title('Survival Rates')
axs[0, 0].set_ylabel('Percentage')
# Grafic pentru distributia claselor pasagerilor
class_counts.plot(kind='bar', ax=axs[0, 1], color='green')
axs[0, 1].set_title('Passenger Class Distribution')
axs[0, 1].set_ylabel('Percentage')
# Grafic pentru distributia genurilor
gender_counts.plot(kind='bar', ax=axs[1, 0], color='purple')
axs[1, 0].set_title('Gender Distribution')
axs[1, 0].set_ylabel('Percentage')
# Aranjez layout-ul graficului pentru a evita suprapunerile
plt.tight_layout()
# Afisez graficul
plt.show()


# TASK 3

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
numeric_columns = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
# Parcurg coloanele numerice si realizez histograma(si pun automat titlurile label-urilor)
for i, col in enumerate(numeric_columns):
    ax = axs[i//3, i%3]
    df[col].hist(ax=ax, bins=10, color='skyblue', edgecolor='black')
    ax.set_title(f'Histogram for {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
axs[1, 2].axis('off')
# Afisez histograma
plt.tight_layout()
plt.show()

# TASK 4

# Calculez numarul de valori lipsa pentru fiecare coloana
missing_data = df.isnull().sum()
missing_columns = missing_data[missing_data > 0]
missing_proportions = (missing_columns / len(df)) * 100

# Afisez coloanele cu valori lipsa si proportiile acestora
print("Coloane cu valori lipsa:")
print(missing_proportions)

# Calculez procentul de valori lipsa pentru fiecare coloana, grupat dupa supravietuire
missing_by_survival = df.groupby('Survived').apply(lambda x: x.isnull().sum() / len(x) * 100)

# Afisez procentul de valori lipsa pentru coloanele selectate, grupat dupa supravietuire
print("\nProcentul de valori lipsa pentru fiecare coloana:")
print(missing_by_survival[missing_columns.index])


# TASK 5

# Creez categoriile de varsta
bins = [0, 20, 40, 60, np.inf]
labels = [0, 1, 2, 3]
df['AgeCategory'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Calculez numarul de pasageri pe categorii de varsta
age_category_counts = df['AgeCategory'].value_counts().sort_index()

# Creez graficul pentru numarul de pasageri pe categorii de varsta
plt.figure(figsize=(10, 6))
age_category_counts.plot(kind='bar', color='teal')
plt.title('Number of Passengers by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Number of Passengers')
plt.xticks(ticks=np.arange(len(labels)), labels=['0-20', '21-40', '41-60', '61+'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# TASK 6

# Calculez numarul de barbati supravietuitori pe categorii de varsta
male_survival = df[(df['Sex'] == 'male') & (df['Survived'] == 1)]['AgeCategory'].value_counts().sort_index()
total_males = df[df['Sex'] == 'male']['AgeCategory'].value_counts().sort_index()
male_survival_rate = (male_survival / total_males) * 100
# Creez o figura pentru a vizualiza rata de supravietuire a barbatilor pe categorii de varsta
plt.figure(figsize=(10, 6))
male_survival_rate.plot(kind='bar', color='orange')
# Adaug titlu si etichete pentru axele x si y
plt.title('Survival Rate of Males by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Survival Rate (%)')
# Setez etichetele pentru axa x
plt.xticks(ticks=np.arange(len(male_survival_rate)), labels=['0-20', '21-40', '41-60', '61+'], rotation=0)
# Adaug o grila pe axa y pentru o vizualizare mai clara
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Afisez graficul
plt.show()


# TASK 7

total_passengers = len(df)
num_children = (df['Age'] < 18).sum()
percent_children = (num_children / total_passengers) * 100
# Calculez rata de supravietuire a copiilor
children_survival_rate = df[df['Age'] < 18]['Survived'].mean() * 100
# Calculez rata de supravietuire a adultilor (18 ani si peste)
adults_survival_rate = df[df['Age'] >= 18]['Survived'].mean() * 100
# Creez o figura pentru a vizualiza rata de supravietuire pe grupe de varsta
plt.figure(figsize=(8, 5))
plt.bar(['Children', 'Adults'], [children_survival_rate, adults_survival_rate], color=['blue', 'green'])
# Adaug titlu si eticheta pentru axa y
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate (%)')
# Setez limitele pentru axa y
plt.ylim(0, 100)
# Adaug o grila pe axa y pentru o vizualizare mai clara
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Afisez graficul
plt.show()
# Printez procentul de copii
print(f"The percent of children: {percent_children:.2f}%")


# TASK 8

# Calculez varsta medie a celor care au supravietuit
mean_age_survived = df[df['Survived'] == 1]['Age'].mean()
# Calculez varsta medie a celor care nu au supravietuit
mean_age_not_survived = df[df['Survived'] == 0]['Age'].mean()
# Inlocuiesc valorile lipsa ale varstei pentru cei care au supravietuit cu varsta medie a celor care au supravietuit
df.loc[(df['Age'].isnull()) & (df['Survived'] == 1), 'Age'] = mean_age_survived
# Inlocuiesc valorile lipsa ale varstei pentru cei care nu au supravietuit cu varsta medie a celor care nu au supravietuit
df.loc[(df['Age'].isnull()) & (df['Survived'] == 0), 'Age'] = mean_age_not_survived
# Determin cea mai comuna valoare pentru coloana 'Embarked'
most_common_embarked = df['Embarked'].mode()[0]
# Inlocuiesc valorile lipsa din coloana 'Embarked' cu cea mai comuna valoare
df['Embarked'].fillna(most_common_embarked, inplace=True)
# Printez numarul de valori lipsa pentru coloana 'Age'
print("The number of absent values for 'Age':", df['Age'].isnull().sum())
print("The number of absent values for 'Embarked':", df['Embarked'].isnull().sum())
# Afisez primele cateva randuri ale DataFrame-ului pentru a verifica modificarile
df.head()


# TASK 9

# Setez titlurile pentru fiecare sex
df['Title'] = df['Name'].apply(lambda name: re.search(r'\b(\w+)\.', name).group(1) if re.search(r'\b(\w+)\.', name) else 'Unknown')
# Calculez numarul de aparitii pentru fiecare titlu
def check_title_sex(row):
    title_sex_mapping = {
        'Mr': 'male',
        'Miss': 'female',
        'Mrs': 'female',
        'Master': 'male',
        'Don': 'male',
        'Dona': 'female',
        'Lady': 'female',
        'Sir': 'male',
        'Countess': 'female',
        'Dr': 'neutral',
        'Rev': 'neutral',
        'Mme': 'female',
        'Ms': 'female',
        'Major': 'male',
        'Capt': 'male',
        'Jonkheer': 'male'
    }
    expected_sex = title_sex_mapping.get(row['Title'], 'neutral')
    if expected_sex == 'neutral':
        return True
    return row['Sex'] == expected_sex
df['Title_Sex_Correct'] = df.apply(check_title_sex, axis=1)
# Calculez numarul de sexe corecte
correct = df['Title_Sex_Correct'].sum()
incorrect = len(df) - correct
print(f"Number of correct titles: {correct}")
print(f"Number of incorrect titles: {incorrect}")
title_counts = df['Title'].value_counts()
# Creez figura pentru a vizualiza distributia titlurilor in functie de numarul de persoane
plt.figure(figsize=(10, 6))
title_counts.plot(kind='bar', color='c')
# Construiesc figura
plt.title('The distribution of titles depending on the number of people')
# Adaug titlul pentru axa x
plt.xlabel('Title')
plt.ylabel('Number of people')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# TASK 10

df['Alone'] = ((df['SibSp'] == 0) & (df['Parch'] == 0)).astype(int)
plt.figure(figsize=(8, 5))
# Desenez o histograma pentru coloana 'Alone', colorata dupa 'Survived'
sns.histplot(data=df, x='Alone', hue='Survived', multiple='stack', discrete=True, palette='pastel')
plt.title('Survival Based on Being Alone on Titanic')
# Adaug titlul pentru axa x
plt.xlabel('Alone (1 = Yes, 0 = No)')
plt.ylabel('Count')
# Setez etichetele axei x
plt.xticks([0, 1], ['Not Alone', 'Alone'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Afisez histograma
plt.show()
plt.figure(figsize=(10, 5))
# Desenez un strip plot pentru primele 100 de inregistrari din DataFrame-ul 'df'
# Adaug dispersia punctelor (jitter=True) si separarea pe categorii (dodge=True)
sns.stripplot(data=df.head(100), x='Pclass', y='Fare', hue='Survived', jitter=True, dodge=True, palette='pastel')
plt.title('Fare vs. Class vs. Survival for the First 100 Records')
plt.xlabel('Passenger Class')
# Adaug eticheta pentru axa y
plt.ylabel('Fare')
plt.grid(True)
plt.show()

