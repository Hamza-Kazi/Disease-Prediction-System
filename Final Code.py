#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from matplotlib import pyplot

from numpy import mean
from imblearn.over_sampling import SMOTE

import warnings 
warnings.filterwarnings('ignore')

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier


# In[ ]:


df = pd.read_csv("heart_2020_cleaned.csv")


# In[ ]:


#df.drop(['SleepTime'], axis=1, inplace=True)
df.drop(['MentalHealth'], axis=1, inplace=True)
df.drop(['Asthma'], axis=1, inplace=True)
df.drop(['AlcoholDrinking'], axis=1, inplace=True)
df.drop(['PhysicalActivity'], axis=1, inplace=True)
df.drop(['Race'], axis =1, inplace = True)
df.drop(['Stroke'], axis = 1, inplace = True)


# In[ ]:


encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                      '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                      '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                      '30-34':32,'25-29':27}
df['AgeCategory'] = df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
df['AgeCategory'] = df['AgeCategory'].astype('float')


# In[ ]:


for col in ['PhysicalHealth', 'SleepTime','BMI','AgeCategory']:#,'MentalHealth'  ]:
    df[col] = df[col]/df[col].max()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
# Integer encode columns with 2 unique values

for col in ['Smoking', 'Diabetic', 'DiffWalking', 'Sex', 'KidneyDisease', 'SkinCancer','HeartDisease']:#, ,'Stroke','PhysicalActivity', 'Asthma', 'AlcoholDrinking'
    if df[col].dtype == 'O':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
# One-hot encode columns with more than 2 unique values
df = pd.get_dummies(df, columns=[ 'GenHealth', ], prefix = ['GenHealth'])#'Race',#'Diabetic',


# In[ ]:


X = df.loc[:, df.columns != 'HeartDisease']
y = df[['HeartDisease']]


# In[ ]:


gb_2 = GradientBoostingClassifier(learning_rate=0.1,max_depth=5,n_estimators=100)
gb_2.fit(X,y)


# In[ ]:


import pickle 
pickle_out = open('classifier_model1.pkl',mode ='wb')  # wb => write in binary mode
pickle.dump(gb_2,pickle_out)  
pickle_out.close()


# # lime 

# In[ ]:


import lime
import numpy
from lime import lime_tabular

explain = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=X.columns,
    class_names=['No','Yes'],
    mode='classification'
)


# In[ ]:


exp = explainer.explain_instance(
    data_row=testX.iloc[141], 
    predict_fn=gb_2.predict_proba,
    num_features = 20
)

exp.show_in_notebook(show_table=True)

