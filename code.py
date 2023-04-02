#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train=pd.read_csv('train.csv.zip')
test=pd.read_csv('test.csv.zip')
store=pd.read_csv('stores.csv')
feature=pd.read_csv('features.csv')


# In[4]:


train.head()


# In[5]:


feature.head()


# In[6]:


store.head()


# In[7]:


df_aux = train.merge(store, how='left', on='Store')
df_train = df_aux.merge(feature, how='left', on=('Store', 'Date', 'IsHoliday'))
df_train.head()


# In[8]:


df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train['Year'] = (df_train.Date.dt.year).astype(int)
df_train['Month'] = (df_train.Date.dt.month).astype(int)
df_train['Week'] = ((df_train.Date.dt.isocalendar().week)*1.0).astype(int)
df_train['Day'] = (df_train.Date.dt.day).astype(int)


# In[9]:


df_train['IsHoliday'].replace({False: 0, True: 1}, inplace=True)


# In[10]:


df_train['SuperBowl'] = df_train['Week'].apply(lambda x: 1 if x == 6 else 0)
df_train['LaborDay'] = df_train['Week'].apply(lambda x: 1 if x == 35 else 0)
df_train['Tranksgiving'] = df_train['Week'].apply(lambda x: 1 if x == 46 else 0)
df_train['Christmas'] = df_train['Week'].apply(lambda x: 1 if x == 51 else 0)
df_train["Tksgiving_to_Xmas"] = df_train['Week'].apply(lambda x: 1 if (x>46) & (x<52)  else 0)


# In[11]:


predictors = ['IsHoliday', 'Month', 'Day', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'SuperBowl', 'LaborDay', 'Tranksgiving', 'Christmas', 'Tksgiving_to_Xmas']


# In[12]:


X_train = df_train[predictors]
y_train = df_train.Weekly_Sales


# In[13]:


def model(model, params, X, Y, cv = None):
    mod = model
    mod_grid = GridSearchCV(mod, params, cv = cv)
    mod_grid.fit(X, Y)
    
    return mod_grid, mod_grid.best_params_


# In[14]:


parameters = {'max_depth': [50,75,100,125],'min_samples_split': [50,75,100,125],'max_features': [0.5,0.75]}
model, best_params = model(DecisionTreeRegressor(), parameters, X_train, y_train, 10)
best_params


# In[15]:


df_train["predict_sales"] = model.predict(df_train[predictors])


# In[16]:


predicted_sales = df_train.groupby('Date')['predict_sales'].sum()
Weekly_Sales = df_train.groupby('Date')['Weekly_Sales'].sum()


# In[17]:


plt.figure(figsize=(20,6))
plt.plot(predicted_sales.values)
plt.plot(Weekly_Sales.values)

plt.yticks( fontsize=16)
plt.ylabel('Sales', fontsize=20, labelpad=20)
plt.xlabel('Weeks', fontsize=20, labelpad=20)

plt.title("Predicts x Real", fontsize=24)
plt.legend(['Decision Tree', 'Real'], fontsize=20);


# In[18]:


def ml_error(df,y,pred,var_feriado):
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))  
    weights = df[var_feriado].apply(lambda x: 1 if x==1 else 5)
    wmae= np.round(np.sum(weights*abs(y-pred))/(np.sum(weights)), 2)
    return pd.DataFrame({'MAE' : mae,'RMSE' : rmse,'WMAE':wmae}, index = [0])


# In[19]:


results = ml_error(X_train, y_train, df_train["predict_sales"], "IsHoliday")
results


# In[20]:


test.head()


# In[21]:


df_aux_test = test.merge(store, how='left', on='Store')
df_test = df_aux_test.merge(feature, how='left', on=('Store', 'Date', 'IsHoliday'))
df_test.head()


# In[22]:


df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test['Year'] = (df_test.Date.dt.year).astype(int)
df_test['Month'] = (df_test.Date.dt.month).astype(int)
df_test['Week'] = ((df_test.Date.dt.isocalendar().week)*1.0).astype(int)
df_test['Day'] = (df_test.Date.dt.day).astype(int)


# In[23]:


df_test['SuperBowl'] = df_test['Week'].apply(lambda x: 1 if x == 6 else 0)
df_test['LaborDay'] = df_test['Week'].apply(lambda x: 1 if x == 35 else 0)
df_test['Tranksgiving'] = df_test['Week'].apply(lambda x: 1 if x == 46 else 0)
df_test['Christmas'] = df_test['Week'].apply(lambda x: 1 if x == 51 else 0)
df_test["Tksgiving_to_Xmas"] = df_test['Week'].apply(lambda x: 1 if (x>46) & (x<52)  else 0)
df_test.head()


# In[24]:


df_test.drop(columns=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], inplace=True)


# In[25]:


df_test.dropna(inplace=True)
df_test["predict_sales"] = model.predict(df_test[predictors])


# In[26]:


predicts = df_test["predict_sales"]
id = df_test['Store'].apply(lambda x: str(x)+'_')+df_test['Dept'].apply(lambda x: str(x)+'_')+df_test['Date'].astype(str)


# In[35]:


sample_submission = pd.concat([id,predicts],axis=1)
sample_submission.head()


# In[31]:


predicted_sales.index=(np.arange(143))


# In[33]:


sale_pred=sample_submission['predict_sales']


# In[34]:


week=2
print('Walmart sales prediction:',end=' ')
print(sale_pred[week])


# In[36]:


d=sample_submission[0]


# In[44]:


date='1_1_2012-11-30'
w=d[d==date].index[0]
print('Walmart sales predicting of given week:',end=' ')
print(sale_pred[w])


# In[45]:


from joblib import dump
dump(model,'wsfp4.joblib')


# In[47]:


import pickle
pickle.dump(model,open('model.sav','wb'))


# In[ ]:




