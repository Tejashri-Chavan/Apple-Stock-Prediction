#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn import linear_model


# In[2]:


df_final=pd.read_csv("AAPL.csv")


# In[3]:


df_final.head()


# In[4]:


df_final.shape


# In[5]:


df_final.describe()


# In[6]:


df_final.isnull().values.any()


# In[7]:


df_final['Adj Close'].plot()


# In[8]:


X=df_final.drop(['Adj Close'],axis=1)
X=X.drop(['Close'],axis=1)


# In[9]:


X.corrwith(df_final['Adj Close']).plot.bar(
          figsize=(20,10), title="Correlation with Adj Close", fontsize=20,
          rot= 90,grid=True)


# In[10]:


test=df_final
target_adj_close=pd.DataFrame(test['Adj Close'])
display(test.head())


# In[11]:


feature_columns=['Open','High','Low','Volume']


# In[12]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
feature_minmax_transform_data=scaler.fit_transform(test[feature_columns])
feature_minmax_transform=pd.DataFrame(columns=feature_columns,data=feature_minmax_transform_data, index=test.index)
feature_minmax_transform.head()


# In[13]:


display(feature_minmax_transform.head())
print('Shape of features :',feature_minmax_transform.shape)
print('Shape of target:', target_adj_close.shape)

target_adj_close=target_adj_close.shift(-1)
validation_y=target_adj_close[-90:-1]
target_adj_close=target_adj_close[:-90]

validation_x=feature_minmax_transform[-90:-1]
feature_minmax_transform=feature_minmax_transform[:-90]
display(validation_x.tail())
display(validation_y.tail())

print("\n-------After process------ \n")
print('Shape of features : ', feature_minmax_transform.shape)
print('Shape of target :',target_adj_close.shape)
display(target_adj_close.tail())


# In[14]:


ts_split = TimeSeriesSplit(n_splits=10)

for train_index, test_index in ts_split.split(feature_minmax_transform):
    X_train = feature_minmax_transform[:len(train_index)]
    X_test = feature_minmax_transform[len(train_index):len(train_index) + len(test_index)]
    y_train = target_adj_close[:len(train_index)].values.ravel()
    y_test = target_adj_close[len(train_index):len(train_index) + len(test_index)]


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


y_train.shape


# In[18]:


y_test.shape


# In[19]:


def validate_result(model,model_name):
    predicted=model.predict(validation_x)
    RSME_score=np.sqrt(mean_squared_error(validation_y,predicted))
    print('RMSE:',RSME_score)
    
    R2_score=r2_score(validation_y,predicted)
    print('R2 score:', R2_score)
    
    plt.plot(validation_y.index,predicted,'r',label='Predict')
    plt.plot(validation_y.index,validation_y,'b',label='Actual')
    plt.ylabel('Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.title(model_name+'Predict vs Actual')
    plt.legend(loc='upper right')
    plt.show()


# In[20]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0)
benchmark_dt=dt.fit(X_train,y_train)
validate_result(benchmark_dt, 'Decision Tree Regression')


# In[21]:


X_train=np.array(X_train)
X_test=np.array(X_test)
X_tr_t=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_tst_t=X_test.reshape(X_test.shape[0],1,X_test.shape[1])


# In[22]:


get_ipython().system('pip install TensorFlow')


# In[23]:


get_ipython().system('pip install Keras')


# In[24]:


import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(1, X_train.shape[1]), activation='relu', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])


# In[26]:


y_pred_test_lstm=model_lstm.predict(X_tst_t)
y_train_pred_lstm=model_lstm.predict(X_tr_t)
print("The R2 score on the Train set is :\t{:0.3f} ".format(r2_score(y_train,y_train_pred_lstm)))
r2_train=r2_score(y_train,y_train_pred_lstm)
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test,y_pred_test_lstm)))
r2_test=r2_score(y_test,y_pred_test_lstm)


# In[29]:


score_lstm=model_lstm.evaluate(X_tst_t,y_test,batch_size=1)


# In[30]:


print('LSTM: %f'%score_lstm)


# In[31]:


y_pred_test_LSTM=model_lstm.predict(X_tst_t)


# In[32]:


plt.plot(y_test,label='True')
plt.plot(y_pred_test_LSTM,label='LSTM')
plt.title("LSTM's_Prediction")
plt.xlabel('Observation')
plt.ylabel('INR_Scaled')
plt.legend()
plt.show()


# In[33]:


col1=pd.DataFrame(y_test,columns=['True'])
col2=pd.DataFrame(y_pred_test_LSTM, columns=['LSTM_prediction'])
col3=pd.DataFrame(history_model_lstm.history['loss'],columns=['Loss_LSTM'])
results=pd.concat([col1,col2,col3],axis=1)
results.to_excel('PredictionResults_LSTM_NonShift.xlsx')


# In[ ]:




