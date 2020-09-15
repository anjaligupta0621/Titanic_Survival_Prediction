#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_test = pd.read_csv('train.csv')

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "No title in the name"
    
    
def shorter_titles(x):
    title = x["Title"]
    if title in ["Capt","Col","Major"]:
        return "Officer"
    elif title in ["Lady","Jonkheer","the Countess","Sir","Don","Dona"]:
        return "Loyalty"
    elif title == "Mme":
        return "Mrs"
    elif title in ["Mlle","Ms"]:
        return "Miss"
    else:
        return title
    
df_test.loc[df_test["Fare"]>400, "Fare"] = df_test["Fare"].median()
df_test.loc[df_test["Age"]>70, "Age"] = 70
df_test.loc[df_test["SibSp"]>5, "SibSp"] = 5

ids = df_test['PassengerId']

df_test["Age"].fillna(df_test["Age"].median(), inplace=True)
df_test["Embarked"].fillna("S", inplace = True)
df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)

del df_test["Cabin"]

df_test['Title'] = df_test['Name'].map(lambda x: get_title(x))
df_test['Title'] = df_test.apply(shorter_titles, axis = 1)

df_test.drop("Name", axis = 1, inplace = True)

df_test.Sex.replace(('male','female'), (0,1), inplace = True)
df_test.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
df_test.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Loyalty','Officer'), (0,1,2,3,4,5,6,7), inplace = True)

del df_test['Ticket']
del df_test['PassengerId']

y = df_test['Survived']
x = df_test.drop(['Survived'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1)
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)

pickle.dump(randomforest, open('titanicmodel.sav', 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




