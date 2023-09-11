from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import extremeboost as xgb
import pandas as pd
import numpy as np

df = pd.read_csv('Telco-customer-churn.csv')
df.drop(['customerID'], axis=1, inplace=True)

df['MultipleLines'].replace(" ", "_", regex=True, inplace=True)
df['tenure'].replace(" ", "_", regex=True, inplace=True)
df['InternetService'].replace(" ", "_", regex=True, inplace=True)
df['OnlineSecurity'].replace(" ", "_", regex=True, inplace=True)
df['StreamingMovies'].replace(" ", "_", regex=True, inplace=True)
df['PaymentMethod'].replace(" ", "_", regex=True, inplace=True)
df['Contract'].replace(" ", "_", regex=True, inplace=True)
df['TechSupport'].replace(" ", "_", regex=True, inplace=True)
df.columns = df.columns.str.replace(' ', '_')

X = df.drop('Churn', axis=1).copy()
print('hooooo')
