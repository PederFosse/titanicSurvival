import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# make it possible to see pandas.pd.head() including all columns
pd.set_option('display.max_columns', 15)

# load passenger data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# concat dataframes
passengers = pd.concat([train, test], ignore_index=True)

# replace 'male' and 'female' with 1 and 0
passengers['Sex'] = passengers['Sex'].map({'female': 1, 'male': 0})

# get the mean age and fill missing age information with mean age
mean_age = passengers['Age'].mean()
passengers['Age'].fillna(value=mean_age, inplace=True)
passengers['Survived'].fillna(value=0, inplace=True)

# create a first and second class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# fetch the relevant features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# list of coefficients with their corresponding columns
coefficient_list = list(zip(['Sex', 'Age', 'FirstClass', 'SecondClass'], model.coef_[0]))
# print(coefficient_list)

# sample passenger creation
Jack = np.array([0.0, 20.0, 0.0, 0.0])      # Male, 20 years, not first class, not second class
Rose = np.array([1.0, 17.0, 1.0, 0.0])      # Female, 17 years, first class, not second class
Peder = np.array([0.0, 22.0, 0.0, 1.0])     # Male, 22 year, not first class, second class

# combine sample passengers into numpy array
sample_passengers = np.array([Jack, Rose, Peder])

# scale sample data
sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)

predictions = model.predict(sample_passengers)

# only rose will survive
print(predictions)
