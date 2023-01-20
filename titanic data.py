#%%
import pandas as pd
# Load Titanic Dataset.
train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')

#%%
# Studying the Titanic Dataset.
train.head()
#%%
train.info()
#%%
# missing values in titanic dataset.
train.isnull().sum()

# Age - 177
# Cabin - 687
# Embarked - 2

#%%
train.describe()
#%%
train['Age'].describe()
#%%
# Visualization.

import matplotlib.pyplot as plt

plt.hist(train['Age'], bins=80)
plt.xlabel('age')
plt.ylabel('count')
plt.show()

plt.hist(train['Age'], bins=8)
plt.xlabel('age')
plt.ylabel('count')
plt.show()

# people in the 20s had the highest count.

#%%
# another method of visualization = seaborn
import seaborn as sns

train['Pclass'] = train['Pclass'].replace(1, "First Class").replace(2, 'Second Class').replace(3, 'Third Class')
train['Survived'] = train['Survived'].replace(0, "Dead").replace(1, "Survived")
sns.countplot(data=train, x='Pclass', hue='Survived')

#%%
ThirdSurvived = train['Fare'].loc[train['Pclass'] == 3].loc[train['Survived']==1]
ThirdSurvived
ThirdDead = train['Fare'].loc[train['Pclass'] == 3].loc[train['Survived']==0]

#%%
ThirdSurvived.mean()
ThirdDead.mean()

#%%
#탑승지에 따라 신규 컬럼 Cherbourg / Southampton / Queenstown Value 
train["Embarked"] = train["Embarked"].replace("C", "Cherbourg").replace("S", "Southampton").replace("Q", "Queenstown")
#탑승지별 생존자수 / 사망자 수를 그래프로 출력
sns.countplot(data=train, x="Embarked", hue="Survived")

# Southampton에서 탑승한 탑승객이 가장 많이 죽었다.

#%%
# visualizing graph by male and female
sns.countplot(data=train, x='Sex', hue='Survived')

#%%
# Data Loading
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning (sklearn)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#%%
import warnings
warnings.filterwarnings(action='ignore')
#%%
# Load Datasets

train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')

# combined data

combine = [train, test]
#%%
# Features

print(train.columns.values)

# PassengerId = Id of the Passenger
# Survived = '1' if survived, '0' if dead.
# Pclass = class of the ticket
#               '1' Upper
#               '2' Middle
#               '3' Lower
# Name = name of the passenger
# Sex = sex of the passenger - male, female
# Age = age of the passenger
# SibSp = number of siblings / spouses aboard the Titanic	
# Parch = number of parents / children aboard the Titanic
# Ticket = ticket number
# Fare = amount of fare for the ticket
# Cabin = cabin number
# Embarked = port of the embarkation
#               'C' = Cherbourg
#               'Q' = Queenstown
#               'S' = Southampton
#%%

# Data

train.head()

train.tail()

# Data information

train.info()

test.info()

#%%
# data analyze

train.describe()

# Survival = 38.4%

# only objects
train.describe(include='O')

# male = 577, female = 314
# most embarked = S, 644

#%%

# Percentage of Survival by Pclass

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Pclass is not set as index if as_index is set to False

# Percentage of Survival is higher by the ticket class

#%%

# Percentage of Survival by Sex

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Percentage of Survival of female is much higher than males

#%%

# Percentage of Survival by SibSp

train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#%%

# Percentage of Survival by Parch

train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#%%

# Distribution of Survival by Age

g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# %%

# Survival by Pclass

grid = sns.FacetGrid(train, col='Survived', row='Pclass', hue='Sex', height=2.2, aspect=1.6)
# width = height * aspect
grid.map(plt.hist, 'Age',alpha =.5, bins=20)
# alpha=int --- 투명도. transparency
grid.add_legend()

# It shows percentage of survival is highest in Pclass = 1
# Children mostly survived in Pclass = 2
# Of all the passengers, most of them did not survive who were Pclass = 3
# Overall, women mostly survived

#%%

# Survival on Embarked location and Pclass

grid1 = sns.FacetGrid(train, row='Embarked', height=2.2, aspect=1.6)
grid1.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order = [1,2,3], hue_order = ["male", "female"])
grid1.add_legend()

# Possibility of higher survival rate of Pclass = 3 than Pclass = 2 which are male Embarked =C, Q

# %%

# Graph showing amount of fare by emabarked, survived, and sex

grid2 = sns.FacetGrid(train, col='Survived', row='Embarked', height=2.2, aspect=1.6)
grid2.map(sns.barplot, 'Sex', 'Fare', alpha =.5, ci=None, order=['male','female'])
grid2.add_legend()

# if Embarked S or C, higher fare has been paid by survived passengers

#%%

# Are we using Ticket and Cabin columns? i don't think so

train = train.drop(['Ticket','Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]

#%%

# Collect titles of passengers in combine dataset to figure out social status

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

#%%

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev','Sir','Jonkheer','Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#%%

# change Titles into knowable integers

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()

#%%

# now delete Name and PassengerId

train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train,test]
train.shape, test.shape

#%%

# also change Sex into knowable integers

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)

train.head()

#%%
